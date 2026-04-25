"""LangChain structured tools that expose DB event data to the LangGraph agent."""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings

logger = structlog.get_logger()


def _get_session() -> Session:
    settings = get_settings()
    engine = create_engine(settings.sync_database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False)
    return SessionLocal()


# ── Tool input schemas ─────────────────────────────────────────────────────────

class GetVideoEventsInput(BaseModel):
    video_id: str = Field(description="UUID of the video to query")
    event_types: list[str] | None = Field(None, description="Filter by event types, e.g. ['shot', 'turnover']")
    min_confidence: float = Field(0.6, description="Minimum confidence threshold")
    limit: int = Field(20, description="Maximum number of events to return")


class GetEventByTimestampInput(BaseModel):
    video_id: str
    timestamp_s: float = Field(description="Timestamp in seconds")
    window_s: float = Field(5.0, description="Search window in seconds around timestamp")


class GetPlayerStatsInput(BaseModel):
    video_id: str
    track_id: int | None = Field(None, description="Specific player track ID, or None for all")


class RetrieveSimilarEventsInput(BaseModel):
    video_id: str
    query_text: str = Field(description="Natural language description of the event to find")
    top_k: int = Field(5, description="Number of similar events to return")


class GetClipSegmentInput(BaseModel):
    video_id: str
    start_s: float
    end_s: float


class SummarizeVideoInput(BaseModel):
    video_id: str


# ── Tool implementations ───────────────────────────────────────────────────────

@tool(args_schema=GetVideoEventsInput)
def get_video_events(
    video_id: str,
    event_types: list[str] | None = None,
    min_confidence: float = 0.6,
    limit: int = 20,
) -> str:
    """Retrieve detected soccer events for a video, optionally filtered by type and confidence."""
    from backend.db.models import EventPrediction

    with _get_session() as session:
        q = (
            select(EventPrediction)
            .where(EventPrediction.video_id == video_id)
            .where(EventPrediction.confidence >= min_confidence)
        )
        if event_types:
            q = q.where(EventPrediction.event_type.in_(event_types))
        q = q.order_by(EventPrediction.start_time_s).limit(limit)
        events = session.execute(q).scalars().all()

    if not events:
        return "No events found matching the criteria."

    lines = [f"Found {len(events)} events:"]
    for e in events:
        ts = _fmt_ts(e.start_time_s)
        track_text = f" track_id={e.nearest_track_id}" if e.nearest_track_id is not None else ""
        lines.append(
            f"  [{ts}] {e.event_type} (conf={e.confidence:.2f}) "
            f"{track_text} event_id={e.id}"
        )
    return "\n".join(lines)


@tool(args_schema=GetEventByTimestampInput)
def get_event_by_timestamp(video_id: str, timestamp_s: float, window_s: float = 5.0) -> str:
    """Find the nearest detected event within a time window around a given timestamp."""
    from backend.db.models import EventPrediction

    with _get_session() as session:
        q = (
            select(EventPrediction)
            .where(EventPrediction.video_id == video_id)
            .where(EventPrediction.start_time_s >= timestamp_s - window_s)
            .where(EventPrediction.start_time_s <= timestamp_s + window_s)
            .order_by(
                # Closest to target timestamp
                (EventPrediction.start_time_s - timestamp_s) * (EventPrediction.start_time_s - timestamp_s)
            )
            .limit(3)
        )
        events = session.execute(q).scalars().all()

    if not events:
        return f"No events found within {window_s}s of timestamp {_fmt_ts(timestamp_s)}"

    lines = []
    for e in events:
        lines.append(
            f"At {_fmt_ts(e.start_time_s)}: {e.event_type} "
            f"(conf={e.confidence:.2f}, track={e.nearest_track_id}, id={e.id})"
        )
    return "\n".join(lines)


@tool(args_schema=GetPlayerStatsInput)
def get_player_stats(video_id: str, track_id: int | None = None) -> str:
    """Retrieve player statistics: possession percentage, events involved, territory zone."""
    from backend.db.models import PlayerStat

    with _get_session() as session:
        q = select(PlayerStat).where(PlayerStat.video_id == video_id)
        if track_id is not None:
            q = q.where(PlayerStat.track_id == track_id)
        q = q.order_by(PlayerStat.total_frames.desc()).limit(20)
        stats = session.execute(q).scalars().all()

    if not stats:
        return "No player stats found."

    lines = [f"Player stats ({len(stats)} players):"]
    for s in stats:
        poss_pct = (s.possession_frames / s.total_frames * 100) if s.total_frames else 0
        lines.append(
            f"  Track {s.track_id}: frames={s.total_frames}, "
            f"possession={poss_pct:.1f}%, zone={s.territory_zone}"
        )
    return "\n".join(lines)


@tool(args_schema=RetrieveSimilarEventsInput)
def retrieve_similar_events(video_id: str, query_text: str, top_k: int = 5) -> str:
    """Find events semantically similar to a text description using pgvector cosine search."""
    settings = get_settings()

    try:
        if settings.openai_api_key:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(input=[query_text], model=settings.embedding_model)
            query_embedding = response.data[0].embedding
        else:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode([query_text])[0]
            if len(emb) < 1536:
                emb = np.pad(emb, (0, 1536 - len(emb)))
            query_embedding = emb.tolist()
    except Exception as exc:
        return f"Embedding failed: {exc}"

    # Format embedding as a Postgres vector literal to avoid SQLAlchemy :: cast conflict
    vec_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    from sqlalchemy import create_engine
    engine = create_engine(settings.sync_database_url)
    try:
        with engine.connect() as conn:
            # Check if any embeddings exist first
            count = conn.execute(
                text("SELECT COUNT(*) FROM event_predictions WHERE video_id = :vid AND embedding IS NOT NULL"),
                {"vid": video_id},
            ).scalar()
            if not count:
                return "No event embeddings available yet. Using keyword-based retrieval instead."

            result = conn.execute(
                text(f"""
                    SELECT id, event_type, confidence, start_time_s, end_time_s, nearest_track_id,
                           1 - (embedding <=> '{vec_literal}'::vector) AS similarity
                    FROM event_predictions
                    WHERE video_id = :video_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> '{vec_literal}'::vector
                    LIMIT :top_k
                """),
                {"video_id": video_id, "top_k": top_k},
            )
            rows = result.fetchall()
    except Exception as exc:
        return f"Similarity search unavailable: {exc}"

    if not rows:
        return "No similar events found (embeddings may not be ready)."

    lines = [f"Top {len(rows)} similar events:"]
    for row in rows:
        ts = _fmt_ts(row.start_time_s)
        lines.append(
            f"  [{ts}] {row.event_type} conf={row.confidence:.2f} "
            f"similarity={row.similarity:.3f} id={row.id}"
        )
    return "\n".join(lines)


@tool(args_schema=GetClipSegmentInput)
def get_clip_segment(video_id: str, start_s: float, end_s: float) -> str:
    """Look up a generated clip covering a time range."""
    from backend.db.models import Clip

    with _get_session() as session:
        q = (
            select(Clip)
            .where(Clip.video_id == video_id)
            .where(Clip.start_time_s <= start_s + 2)
            .where(Clip.end_time_s >= end_s - 2)
            .limit(1)
        )
        clip = session.execute(q).scalar_one_or_none()

    if not clip:
        return f"No clip found covering {_fmt_ts(start_s)}-{_fmt_ts(end_s)}"
    return f"Clip available: {clip.file_path} ({_fmt_ts(clip.start_time_s)}-{_fmt_ts(clip.end_time_s)})"


@tool(args_schema=SummarizeVideoInput)
def summarize_video(video_id: str) -> str:
    """Get a high-level summary: event counts by type, top events, player possession stats."""
    from backend.db.models import EventPrediction, PlayerStat
    from sqlalchemy import func

    with _get_session() as session:
        # Event type counts
        count_q = (
            select(EventPrediction.event_type, func.count().label("n"))
            .where(EventPrediction.video_id == video_id)
            .group_by(EventPrediction.event_type)
        )
        counts = session.execute(count_q).all()

        # Top 5 by confidence
        top_q = (
            select(EventPrediction)
            .where(EventPrediction.video_id == video_id)
            .order_by(EventPrediction.confidence.desc())
            .limit(5)
        )
        top_events = session.execute(top_q).scalars().all()

        # Top players by possession
        player_q = (
            select(PlayerStat)
            .where(PlayerStat.video_id == video_id)
            .order_by(PlayerStat.possession_frames.desc())
            .limit(5)
        )
        players = session.execute(player_q).scalars().all()

    lines = ["=== Video Summary ==="]
    lines.append("Event counts:")
    for row in counts:
        lines.append(f"  {row.event_type}: {row.n}")

    lines.append("\nTop events by confidence:")
    for e in top_events:
        lines.append(f"  [{_fmt_ts(e.start_time_s)}] {e.event_type} conf={e.confidence:.2f}")

    lines.append("\nTop players by possession:")
    for p in players:
        pct = (p.possession_frames / p.total_frames * 100) if p.total_frames else 0
        lines.append(f"  Track {p.track_id}: {pct:.1f}% possession, zone={p.territory_zone}")

    return "\n".join(lines)


def get_all_tools():
    return [
        get_video_events,
        get_event_by_timestamp,
        get_player_stats,
        retrieve_similar_events,
        get_clip_segment,
        summarize_video,
    ]


def _fmt_ts(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
