"""Main Celery task: orchestrates the full video processing pipeline."""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
import shutil

import structlog
from celery import Task
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings
from backend.workers.celery_app import celery_app

logger = structlog.get_logger()


def _get_sync_session() -> sessionmaker:
    settings = get_settings()
    engine = create_engine(settings.sync_database_url, pool_pre_ping=True)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class VideoProcessingTask(Task):
    _session_factory = None

    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = _get_sync_session()
        return self._session_factory


@celery_app.task(
    bind=True,
    base=VideoProcessingTask,
    name="backend.workers.process_video_job.process_video",
    max_retries=2,
    default_retry_delay=30,
)
def process_video(self: VideoProcessingTask, video_id: str, job_id: str) -> dict:
    settings = get_settings()
    SessionLocal = self.session_factory

    def _update_job(session: Session, step: str, progress: int):
        from backend.db.models import ProcessingJob
        job = session.get(ProcessingJob, job_id)
        if job:
            job.current_step = step
            job.progress = progress
            session.commit()
        logger.info("job.progress", job_id=job_id, step=step, progress=progress)

    def _fail_job(session: Session, error: str):
        from backend.db.models import ProcessingJob, Video
        job = session.get(ProcessingJob, job_id)
        if job:
            job.status = "failed"
            job.error_msg = error
            job.finished_at = datetime.now(timezone.utc)
            session.commit()
        video = session.get(Video, video_id)
        if video:
            video.status = "failed"
            session.commit()

    with SessionLocal() as session:
        try:
            from backend.db.models import ProcessingJob, Video

            # ── Mark job started ──────────────────────────────────────────
            job = session.get(ProcessingJob, job_id)
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            session.commit()

            video = session.get(Video, video_id)
            if not video or not video.file_path:
                raise ValueError(f"Video {video_id} not found or missing file_path")

            video_path = Path(video.file_path)
            processed_dir = settings.processed_dir / video_id
            features_dir = settings.features_dir / video_id
            clips_dir = settings.clips_dir / video_id

            # A re-run should replace derived artifacts, not append duplicate rows.
            from backend.db.models import Clip, Detection, EventPrediction, Frame, PlayerStat, Track

            session.execute(delete(Clip).where(Clip.video_id == video_id))
            session.execute(delete(PlayerStat).where(PlayerStat.video_id == video_id))
            session.execute(delete(EventPrediction).where(EventPrediction.video_id == video_id))
            session.execute(delete(Track).where(Track.video_id == video_id))
            frame_ids = [
                row[0]
                for row in session.execute(
                    select(Frame.id).where(Frame.video_id == video_id)
                ).all()
            ]
            if frame_ids:
                session.execute(delete(Detection).where(Detection.frame_id.in_(frame_ids)))
                session.execute(delete(Frame).where(Frame.id.in_(frame_ids)))
            session.commit()

            shutil.rmtree(processed_dir, ignore_errors=True)
            shutil.rmtree(features_dir, ignore_errors=True)
            shutil.rmtree(clips_dir, ignore_errors=True)

            # ── Step 1: Extract frames ────────────────────────────────────
            _update_job(session, "extract_frames", 5)
            from cv.extract_frames import extract_frames
            frames_dir = processed_dir / "frames"
            meta, extracted = extract_frames(
                video_path=video_path,
                output_dir=frames_dir,
                sample_rate=settings.frame_sample_rate,
            )
            video.fps = meta.fps
            video.width = meta.width
            video.height = meta.height
            video.total_frames = meta.total_frames
            video.duration_s = meta.duration_s
            session.commit()

            # Insert frame records (bulk)
            frame_objects = [
                Frame(
                    video_id=video_id,
                    frame_index=f.frame_index,
                    timestamp_s=f.timestamp_s,
                    file_path=f.file_path,
                )
                for f in extracted
            ]
            session.bulk_save_objects(frame_objects)
            session.commit()

            # Re-fetch frames to get IDs
            frame_rows = session.execute(
                select(Frame).where(Frame.video_id == video_id).order_by(Frame.frame_index)
            ).scalars().all()
            frame_id_map = {f.frame_index: f.id for f in frame_rows}

            # ── Step 2: Detect objects ────────────────────────────────────
            _update_job(session, "detect_objects", 20)
            from cv.detect_objects import run_detection
            frame_tuples = [(f.frame_index, f.timestamp_s, f.file_path) for f in extracted]

            last_detection_progress = {"value": 20, "done": 0}

            def _update_detection_progress(done: int, total: int) -> None:
                if total <= 0:
                    return
                progress = 20 + int((done / total) * 14)
                batch_delta = done - last_detection_progress["done"]
                if progress > last_detection_progress["value"] or batch_delta >= 160 or done == total:
                    last_detection_progress["value"] = progress
                    last_detection_progress["done"] = done
                    _update_job(session, f"detect_objects ({done}/{total} frames)", progress)

            detections = run_detection(
                frame_paths=frame_tuples,
                model_name=settings.yolo_model,
                confidence_threshold=0.3,
                progress_callback=_update_detection_progress,
            )

            # ── Step 3: Track objects ─────────────────────────────────────
            _update_job(session, "track_objects", 35)
            from cv.track_objects import run_tracking
            detections, track_summaries = run_tracking(detections, meta.width, meta.height)

            # Bulk insert detections
            from backend.db.models import Detection
            det_objects = [
                Detection(
                    frame_id=frame_id_map.get(d.frame_index),
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox_x1=d.bbox_x1,
                    bbox_y1=d.bbox_y1,
                    bbox_x2=d.bbox_x2,
                    bbox_y2=d.bbox_y2,
                    track_id=d.track_id,
                )
                for d in detections
                if frame_id_map.get(d.frame_index) is not None
            ]
            session.bulk_save_objects(det_objects)

            # Insert track summaries
            from backend.db.models import Track
            track_objects = [
                Track(
                    video_id=video_id,
                    track_id=t.track_id,
                    class_name=t.class_name,
                    first_seen_s=t.first_seen_s,
                    last_seen_s=t.last_seen_s,
                    frame_count=t.frame_count,
                )
                for t in track_summaries
            ]
            session.bulk_save_objects(track_objects)
            session.commit()

            # ── Step 4: Generate features ────────────────────────────────
            _update_job(session, "generate_features", 50)
            from cv.estimate_possession import estimate_possession
            from cv.generate_features import build_feature_matrix, save_features
            possession_ests = estimate_possession(detections)
            features, frame_meta = build_feature_matrix(
                detections, possession_ests, meta.width, meta.height, meta.fps
            )
            save_features(features, frame_meta, features_dir)

            # ── Step 5: Classify events ──────────────────────────────────
            _update_job(session, "classify_events", 65)
            from backend.services.event_service import classify_and_store_events
            classify_and_store_events(
                session=session,
                video_id=video_id,
                features=features,
                frame_meta=frame_meta,
                fps=meta.fps,
                confidence_threshold=settings.event_confidence_threshold,
            )

            # ── Step 6: Embed events ─────────────────────────────────────
            _update_job(session, "embed_events", 78)
            from backend.services.embedding_service import embed_video_events
            embed_video_events(session=session, video_id=video_id)

            # ── Step 7: Aggregate player stats ───────────────────────────
            _update_job(session, "aggregate_stats", 88)
            from backend.services.video_processor import aggregate_player_stats
            aggregate_player_stats(
                session=session,
                video_id=video_id,
                detections=detections,
                possession_estimates=possession_ests,
            )

            # ── Step 8: Generate clips ────────────────────────────────────
            _update_job(session, "generate_clips", 93)
            from backend.services.clip_service import generate_event_clips
            generate_event_clips(
                session=session,
                video_id=video_id,
                video_path=str(video_path),
                clips_dir=str(settings.clips_dir / video_id),
                top_n=20,
            )

            # ── Mark done ────────────────────────────────────────────────
            job.status = "done"
            job.progress = 100
            job.current_step = "complete"
            job.finished_at = datetime.now(timezone.utc)
            video.status = "done"
            session.commit()

            logger.info("job.done", job_id=job_id, video_id=video_id)
            return {"status": "done", "video_id": video_id, "job_id": job_id}

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("job.failed", job_id=job_id, video_id=video_id, error=str(exc), traceback=tb)
            _fail_job(session, str(exc))
            raise
