import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.responses import APIEnvelope, ok
from backend.db.database import get_db
from backend.db.models import Video, AgentRun

router = APIRouter()
logger = structlog.get_logger()


class AskRequest(BaseModel):
    question: str


class EvidenceItem(BaseModel):
    event_id: str
    event_type: str
    start_time_s: float
    end_time_s: float
    confidence: float
    nearest_track_id: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    intent: str | None
    evidence: list[EvidenceItem]
    latency_ms: int
    run_id: str


@router.post("/{video_id}/ask", response_model=APIEnvelope)
async def ask_question(
    video_id: str,
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
):
    result_v = await db.execute(select(Video).where(Video.id == video_id))
    video = result_v.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(status_code=409, detail="Video processing not complete")

    from backend.agents.film_room_graph import run_film_room_agent

    start = time.monotonic()
    run_id = str(uuid.uuid4())

    try:
        result = await run_film_room_agent(
            video_id=video_id,
            question=body.question,
            run_id=run_id,
        )
    except Exception as exc:
        logger.error("agent.error", video_id=video_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    latency_ms = int((time.monotonic() - start) * 1000)

    agent_run = AgentRun(
        id=run_id,
        video_id=video_id,
        question=body.question,
        answer=result["answer"],
        intent=result.get("intent"),
        events_used=[e["event_id"] for e in result.get("evidence", [])],
        latency_ms=latency_ms,
    )
    db.add(agent_run)
    await db.commit()

    return ok(
        AskResponse(
            question=body.question,
            answer=result["answer"],
            intent=result.get("intent"),
            evidence=[EvidenceItem(**e) for e in result.get("evidence", [])],
            latency_ms=latency_ms,
            run_id=run_id,
        )
    )
