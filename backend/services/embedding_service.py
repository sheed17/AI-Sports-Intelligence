"""Generate text embeddings for event predictions and store in pgvector column."""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.db.models import EventPrediction

logger = structlog.get_logger()


def _build_event_text(event: EventPrediction) -> str:
    """Construct a human-readable description of an event for embedding."""
    ts_start = f"{event.start_time_s:.1f}s"
    ts_end = f"{event.end_time_s:.1f}s"
    meta = event.metadata_ or {}
    top3 = meta.get("top3", [])
    top3_str = ", ".join(f"{cls}:{conf:.2f}" for cls, conf in top3[:3]) if top3 else ""

    parts = [
        f"Soccer event: {event.event_type}",
        f"Time: {ts_start} to {ts_end}",
        f"Confidence: {event.confidence:.2f}",
    ]
    if event.possession_team:
        parts.append(f"Possession: {event.possession_team}")
    if event.nearest_track_id is not None:
        parts.append(f"Player ID: {event.nearest_track_id}")
    if top3_str:
        parts.append(f"Top predictions: {top3_str}")
    return ". ".join(parts)


def _embed_texts_openai(texts: list[str], model: str) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return [r.embedding for r in response.data]


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    # Pad/truncate to 1536 dims to match pgvector column
    import numpy as np
    result = []
    for emb in embeddings:
        if len(emb) < 1536:
            emb = np.pad(emb, (0, 1536 - len(emb)))
        else:
            emb = emb[:1536]
        result.append(emb.tolist())
    return result


def embed_video_events(session: Session, video_id: str, batch_size: int = 50) -> int:
    """Embed all event predictions for a video and update the embedding column."""
    settings = get_settings()

    result = session.execute(
        select(EventPrediction)
        .where(EventPrediction.video_id == video_id)
        .where(EventPrediction.embedding.is_(None))
    )
    events = result.scalars().all()
    if not events:
        return 0

    texts = [_build_event_text(e) for e in events]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            if settings.openai_api_key:
                batch_embs = _embed_texts_openai(batch, settings.embedding_model)
            else:
                batch_embs = _embed_texts_local(batch)
            all_embeddings.extend(batch_embs)
        except Exception as exc:
            logger.warning("embedding.batch_failed", error=str(exc), using_local=True)
            all_embeddings.extend(_embed_texts_local(batch))

    for event, embedding in zip(events, all_embeddings):
        event.embedding = embedding

    session.commit()
    logger.info("embedding.done", video_id=video_id, count=len(events))
    return len(events)
