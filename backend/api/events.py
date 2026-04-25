from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.responses import APIEnvelope, ok
from backend.db.database import get_db
from backend.db.models import EventPrediction, Video

router = APIRouter()


class EventResponse(BaseModel):
    id: str
    video_id: str
    event_type: str
    confidence: float
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int
    possession_team: str | None
    nearest_track_id: int | None
    metadata: dict[str, Any] | None
    created_at: str

    class Config:
        from_attributes = True


@router.get("/{video_id}/events", response_model=APIEnvelope)
async def list_events(
    video_id: str,
    event_type: str | None = Query(None, description="Filter by event type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    await _require_video(video_id, db)

    q = select(EventPrediction).where(
        EventPrediction.video_id == video_id,
        EventPrediction.confidence >= min_confidence,
    )
    if event_type:
        q = q.where(EventPrediction.event_type == event_type)
    q = q.order_by(EventPrediction.start_time_s).offset(offset).limit(limit)

    result = await db.execute(q)
    events = result.scalars().all()
    data = [_to_response(e) for e in events]
    return ok(data, meta={"count": len(data), "limit": limit, "offset": offset})


@router.get("/{video_id}/events/{event_id}", response_model=APIEnvelope)
async def get_event(video_id: str, event_id: str, db: AsyncSession = Depends(get_db)):
    await _require_video(video_id, db)
    result = await db.execute(
        select(EventPrediction).where(
            EventPrediction.id == event_id,
            EventPrediction.video_id == video_id,
        )
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return ok(_to_response(event))


def _to_response(e: EventPrediction) -> EventResponse:
    return EventResponse(
        id=e.id,
        video_id=e.video_id,
        event_type=e.event_type,
        confidence=e.confidence,
        start_time_s=e.start_time_s,
        end_time_s=e.end_time_s,
        start_frame=e.start_frame,
        end_frame=e.end_frame,
        possession_team=e.possession_team,
        nearest_track_id=e.nearest_track_id,
        metadata=e.metadata_,
        created_at=e.created_at.isoformat(),
    )


async def _require_video(video_id: str, db: AsyncSession) -> None:
    result = await db.execute(select(Video).where(Video.id == video_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Video not found")
