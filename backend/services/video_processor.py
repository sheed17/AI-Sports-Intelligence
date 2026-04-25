"""Aggregate player statistics after detection + tracking is complete."""
from __future__ import annotations

import uuid
from collections import defaultdict

import structlog
from sqlalchemy.orm import Session

from backend.db.models import PlayerStat
from cv.detect_objects import FrameDetection
from cv.estimate_possession import PossessionEstimate

logger = structlog.get_logger()


def aggregate_player_stats(
    session: Session,
    video_id: str,
    detections: list[FrameDetection],
    possession_estimates: list[PossessionEstimate],
) -> None:
    """Compute per-track statistics and insert into player_stats table."""
    frame_counts: dict[int, int] = defaultdict(int)
    possession_frames: dict[int, int] = defaultdict(int)
    velocity_sums: dict[int, list[float]] = defaultdict(list)
    x_positions: dict[int, list[float]] = defaultdict(list)

    for det in detections:
        if det.class_name != "player" or det.track_id is None:
            continue
        frame_counts[det.track_id] += 1
        cx = (det.bbox_x1 + det.bbox_x2) / 2
        x_positions[det.track_id].append(cx)

    for poss in possession_estimates:
        if poss.possessing_track_id is not None:
            possession_frames[poss.possessing_track_id] += 1

    stats = []
    for track_id, count in frame_counts.items():
        xs = x_positions[track_id]
        avg_x = sum(xs) / len(xs) if xs else 0.0
        # Rough territory heuristic (assumes frame width normalized elsewhere)
        if avg_x < 0.33:
            zone = "left"
        elif avg_x < 0.67:
            zone = "center"
        else:
            zone = "right"

        stats.append(
            PlayerStat(
                id=str(uuid.uuid4()),
                video_id=video_id,
                track_id=track_id,
                total_frames=count,
                possession_frames=possession_frames.get(track_id, 0),
                events_involved={},
                avg_velocity=None,
                territory_zone=zone,
            )
        )

    session.bulk_save_objects(stats)
    session.commit()
    logger.info("player_stats.aggregated", video_id=video_id, players=len(stats))
