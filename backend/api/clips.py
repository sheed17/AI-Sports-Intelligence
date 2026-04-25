from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.responses import APIEnvelope, ok
from backend.db.database import get_db
from backend.db.models import Clip, Video

router = APIRouter()


class ClipResponse(BaseModel):
    id: str
    video_id: str
    event_id: str | None
    file_path: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    created_at: str


@router.get("/videos/{video_id}/clips", response_model=APIEnvelope)
async def list_clips(video_id: str, db: AsyncSession = Depends(get_db)):
    result_v = await db.execute(select(Video).where(Video.id == video_id))
    if not result_v.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(
        select(Clip).where(Clip.video_id == video_id).order_by(Clip.start_time_s)
    )
    clips = result.scalars().all()
    data = [_to_response(c) for c in clips]
    return ok(data, meta={"count": len(data)})


@router.get("/clips/{clip_id}/stream")
async def stream_clip(clip_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Clip).where(Clip.id == clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    path = Path(clip.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip file not found on disk")

    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


def _to_response(c: Clip) -> ClipResponse:
    return ClipResponse(
        id=c.id,
        video_id=c.video_id,
        event_id=c.event_id,
        file_path=c.file_path,
        start_time_s=c.start_time_s,
        end_time_s=c.end_time_s,
        duration_s=c.duration_s,
        created_at=c.created_at.isoformat(),
    )
