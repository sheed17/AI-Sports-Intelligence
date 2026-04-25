import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger, Boolean, Float, ForeignKey, Integer, String, Text,
    TIMESTAMP, func, JSON,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from backend.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    total_frames: Mapped[int | None] = mapped_column(Integer)
    # uploaded | processing | done | failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    jobs: Mapped[list["ProcessingJob"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    frames: Mapped[list["Frame"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    events: Mapped[list["EventPrediction"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    clips: Mapped[list["Clip"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    player_stats: Mapped[list["PlayerStat"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    agent_runs: Mapped[list["AgentRun"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id", ondelete="CASCADE"))
    celery_task_id: Mapped[str | None] = mapped_column(Text)
    # pending | running | done | failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    current_step: Mapped[str | None] = mapped_column(Text)
    error_msg: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship(back_populates="jobs")


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id", ondelete="CASCADE"))
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_s: Mapped[float] = mapped_column(Float, nullable=False)
    file_path: Mapped[str | None] = mapped_column(Text)

    video: Mapped["Video"] = relationship(back_populates="frames")
    detections: Mapped[list["Detection"]] = relationship(back_populates="frame", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    frame_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("frames.id", ondelete="CASCADE"))
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)  # player|ball|referee|goalpost
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float | None] = mapped_column(Float)
    bbox_y1: Mapped[float | None] = mapped_column(Float)
    bbox_x2: Mapped[float | None] = mapped_column(Float)
    bbox_y2: Mapped[float | None] = mapped_column(Float)
    track_id: Mapped[int | None] = mapped_column(Integer)  # assigned by ByteTrack

    frame: Mapped["Frame"] = relationship(back_populates="detections")


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen_s: Mapped[float | None] = mapped_column(Float)
    last_seen_s: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)


class EventPrediction(Base):
    __tablename__ = "event_predictions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id", ondelete="CASCADE"))
    # pass | shot | dribble | turnover | save | buildup
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    start_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    possession_team: Mapped[str | None] = mapped_column(String(32))
    nearest_track_id: Mapped[int | None] = mapped_column(Integer)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship(back_populates="events")
    clip: Mapped["Clip | None"] = relationship(back_populates="event", uselist=False)


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id", ondelete="CASCADE"))
    event_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("event_predictions.id"))
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    start_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship(back_populates="clips")
    event: Mapped["EventPrediction | None"] = relationship(back_populates="clip")


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    possession_frames: Mapped[int] = mapped_column(Integer, default=0)
    events_involved: Mapped[dict | None] = mapped_column(JSONB)  # {pass: 3, shot: 1, ...}
    avg_velocity: Mapped[float | None] = mapped_column(Float)
    territory_zone: Mapped[str | None] = mapped_column(String(32))  # left|center|right

    video: Mapped["Video"] = relationship(back_populates="player_stats")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("videos.id"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(64))
    events_used: Mapped[list | None] = mapped_column(JSONB)
    langsmith_run_id: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    video: Mapped["Video | None"] = relationship(back_populates="agent_runs")
