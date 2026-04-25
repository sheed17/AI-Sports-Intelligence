"""Generate short MP4 clips around detected events using FFmpeg."""
from __future__ import annotations

import uuid
from pathlib import Path

import ffmpeg
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Clip, EventPrediction

logger = structlog.get_logger()

CLIP_PAD_S = 1.5  # seconds of padding before and after event


def generate_event_clips(
    session: Session,
    video_id: str,
    video_path: str,
    clips_dir: str,
    top_n: int = 20,
) -> int:
    """Generate clips for the top_n highest-confidence events.

    Returns number of clips created.
    """
    clips_path = Path(clips_dir)
    clips_path.mkdir(parents=True, exist_ok=True)

    existing_event_ids = {
        row[0]
        for row in session.execute(
            select(Clip.event_id)
            .where(Clip.video_id == video_id)
            .where(Clip.event_id.is_not(None))
        ).all()
    }

    result = session.execute(
        select(EventPrediction)
        .where(EventPrediction.video_id == video_id)
        .order_by(EventPrediction.confidence.desc())
        .limit(top_n)
    )
    events = result.scalars().all()
    if not events:
        logger.info("clip_service.no_events", video_id=video_id)
        return 0

    created = 0
    for event in events:
        if event.id in existing_event_ids:
            continue

        start_s = max(0.0, event.start_time_s - CLIP_PAD_S)
        end_s = event.end_time_s + CLIP_PAD_S
        duration_s = end_s - start_s
        clip_id = str(uuid.uuid4())
        out_path = clips_path / f"{clip_id}.mp4"

        try:
            (
                ffmpeg.input(video_path, ss=start_s, t=duration_s)
                .output(
                    str(out_path),
                    vcodec="libx264",
                    preset="fast",
                    crf=23,
                    acodec="aac",
                    movflags="+faststart",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            clip = Clip(
                id=clip_id,
                video_id=video_id,
                event_id=event.id,
                file_path=str(out_path),
                start_time_s=start_s,
                end_time_s=end_s,
                duration_s=duration_s,
            )
            session.add(clip)
            created += 1
        except ffmpeg.Error as exc:
            logger.error("clip_service.ffmpeg_error", event_id=event.id, error=str(exc))

    session.commit()
    logger.info("clip_service.done", video_id=video_id, clips_created=created)
    return created
