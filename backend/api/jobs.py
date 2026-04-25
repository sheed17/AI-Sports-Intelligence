from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.responses import APIEnvelope, ok
from backend.db.database import get_db
from backend.db.models import ProcessingJob

router = APIRouter()


class JobResponse(BaseModel):
    id: str
    video_id: str
    celery_task_id: str | None
    status: str
    progress: int
    current_step: str | None
    error_msg: str | None
    started_at: str | None
    finished_at: str | None
    created_at: str


@router.get("/{job_id}", response_model=APIEnvelope)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return ok(
        JobResponse(
            id=job.id,
            video_id=job.video_id,
            celery_task_id=job.celery_task_id,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
            error_msg=job.error_msg,
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            created_at=job.created_at.isoformat(),
        )
    )
