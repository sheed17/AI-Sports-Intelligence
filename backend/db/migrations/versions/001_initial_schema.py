"""Initial schema with all tables and pgvector

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("original_filename", sa.Text),
        sa.Column("file_path", sa.Text),
        sa.Column("duration_s", sa.Float),
        sa.Column("fps", sa.Float),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("total_frames", sa.Integer),
        sa.Column("status", sa.String(32), nullable=False, server_default="uploaded"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id", ondelete="CASCADE")),
        sa.Column("celery_task_id", sa.Text),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Integer, server_default="0"),
        sa.Column("current_step", sa.Text),
        sa.Column("error_msg", sa.Text),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "frames",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id", ondelete="CASCADE")),
        sa.Column("frame_index", sa.Integer, nullable=False),
        sa.Column("timestamp_s", sa.Float, nullable=False),
        sa.Column("file_path", sa.Text),
    )
    op.create_index("ix_frames_video_id", "frames", ["video_id"])
    op.create_index("ix_frames_timestamp_s", "frames", ["timestamp_s"])

    op.create_table(
        "detections",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("frame_id", sa.BigInteger, sa.ForeignKey("frames.id", ondelete="CASCADE")),
        sa.Column("class_name", sa.String(64), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("bbox_x1", sa.Float),
        sa.Column("bbox_y1", sa.Float),
        sa.Column("bbox_x2", sa.Float),
        sa.Column("bbox_y2", sa.Float),
        sa.Column("track_id", sa.Integer),
    )
    op.create_index("ix_detections_frame_id", "detections", ["frame_id"])
    op.create_index("ix_detections_track_id", "detections", ["track_id"])

    op.create_table(
        "tracks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id", ondelete="CASCADE")),
        sa.Column("track_id", sa.Integer, nullable=False),
        sa.Column("class_name", sa.String(64), nullable=False),
        sa.Column("first_seen_s", sa.Float),
        sa.Column("last_seen_s", sa.Float),
        sa.Column("frame_count", sa.Integer, server_default="0"),
    )

    op.create_table(
        "event_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id", ondelete="CASCADE")),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("start_time_s", sa.Float, nullable=False),
        sa.Column("end_time_s", sa.Float, nullable=False),
        sa.Column("start_frame", sa.Integer, nullable=False),
        sa.Column("end_frame", sa.Integer, nullable=False),
        sa.Column("possession_team", sa.String(32)),
        sa.Column("nearest_track_id", sa.Integer),
        sa.Column("metadata", postgresql.JSONB),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    # Add vector column using raw SQL (pgvector type not in core SQLAlchemy DDL)
    op.execute("ALTER TABLE event_predictions ADD COLUMN embedding vector(1536)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_event_embeddings "
        "ON event_predictions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)"
    )
    op.create_index("ix_events_video_id", "event_predictions", ["video_id"])
    op.create_index("ix_events_start_time", "event_predictions", ["start_time_s"])
    op.create_index("ix_events_event_type", "event_predictions", ["event_type"])

    op.create_table(
        "clips",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id", ondelete="CASCADE")),
        sa.Column("event_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("event_predictions.id")),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("start_time_s", sa.Float, nullable=False),
        sa.Column("end_time_s", sa.Float, nullable=False),
        sa.Column("duration_s", sa.Float, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "player_stats",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id", ondelete="CASCADE")),
        sa.Column("track_id", sa.Integer, nullable=False),
        sa.Column("total_frames", sa.Integer, server_default="0"),
        sa.Column("possession_frames", sa.Integer, server_default="0"),
        sa.Column("events_involved", postgresql.JSONB),
        sa.Column("avg_velocity", sa.Float),
        sa.Column("territory_zone", sa.String(32)),
    )

    op.create_table(
        "agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("videos.id")),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("intent", sa.String(64)),
        sa.Column("events_used", postgresql.JSONB),
        sa.Column("langsmith_run_id", sa.Text),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("agent_runs")
    op.drop_table("player_stats")
    op.drop_table("clips")
    op.drop_table("event_predictions")
    op.drop_table("tracks")
    op.drop_table("detections")
    op.drop_table("frames")
    op.drop_table("processing_jobs")
    op.drop_table("videos")
