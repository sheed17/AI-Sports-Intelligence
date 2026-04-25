import uuid
import shutil

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings, Settings
from backend.api.responses import APIEnvelope, ok
from backend.db.database import get_db
from backend.db.models import Video, ProcessingJob
from backend.workers.celery_app import celery_app

router = APIRouter()
logger = structlog.get_logger()


class VideoResponse(BaseModel):
    id: str
    filename: str
    status: str
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None
    total_frames: int | None
    created_at: str

    class Config:
        from_attributes = True


class ProcessResponse(BaseModel):
    video_id: str
    job_id: str
    celery_task_id: str


def _video_response(video: Video) -> VideoResponse:
    return VideoResponse(
        id=video.id,
        filename=video.filename,
        status=video.status,
        duration_s=video.duration_s,
        fps=video.fps,
        width=video.width,
        height=video.height,
        total_frames=video.total_frames,
        created_at=video.created_at.isoformat(),
    )


@router.post("/upload", response_model=APIEnvelope, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    video_id = str(uuid.uuid4())
    dest_dir = settings.raw_videos_dir / video_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_mb = dest_path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.max_video_size_mb:
        dest_path.unlink()
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_video_size_mb}MB limit")

    video = Video(
        id=video_id,
        filename=file.filename,
        original_filename=file.filename,
        file_path=str(dest_path),
        status="uploaded",
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    logger.info("video.uploaded", video_id=video_id, filename=file.filename, size_mb=round(file_size_mb, 2))

    return ok(_video_response(video))


@router.get("", response_model=APIEnvelope)
async def list_videos(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    q = select(Video)
    if status_filter:
        q = q.where(Video.status == status_filter)
    q = q.order_by(Video.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(q)
    videos = [_video_response(video) for video in result.scalars().all()]
    return ok(videos, meta={"count": len(videos), "limit": limit, "offset": offset})


@router.get("/{video_id}", response_model=APIEnvelope)
async def get_video(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await _get_video_or_404(video_id, db)
    return ok(_video_response(video))


@router.get("/{video_id}/status")
async def get_video_status(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await _get_video_or_404(video_id, db)
    result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.video_id == video_id)
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return ok({
        "video_id": video_id,
        "video_status": video.status,
        "job": {
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "current_step": job.current_step,
            "error_msg": job.error_msg,
        } if job else None,
    })


@router.post("/{video_id}/process", response_model=APIEnvelope)
async def process_video(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await _get_video_or_404(video_id, db)
    if video.status in ("processing",):
        raise HTTPException(status_code=409, detail="Video is already being processed")

    job_id = str(uuid.uuid4())
    job = ProcessingJob(id=job_id, video_id=video_id, status="pending")
    db.add(job)

    video.status = "processing"
    await db.commit()

    task = celery_app.send_task(
        "backend.workers.process_video_job.process_video",
        args=[video_id, job_id],
        task_id=job_id,
    )

    job.celery_task_id = task.id
    await db.commit()

    logger.info("video.process_enqueued", video_id=video_id, job_id=job_id, task_id=task.id)
    return ok(ProcessResponse(video_id=video_id, job_id=job_id, celery_task_id=task.id))


async def _get_video_or_404(video_id: str, db: AsyncSession) -> Video:
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video
