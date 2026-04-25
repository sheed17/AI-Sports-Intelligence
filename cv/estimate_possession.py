"""Estimate ball possession from per-frame detections using nearest-player heuristic."""
from __future__ import annotations

import math
from dataclasses import dataclass

from cv.detect_objects import FrameDetection


@dataclass
class PossessionEstimate:
    frame_index: int
    timestamp_s: float
    possessing_track_id: int | None   # track_id of nearest player
    distance_px: float                # distance in pixels
    n_players_near_ball: int          # within NEAR_BALL_THRESHOLD


# Player is "near ball" if within this many pixels of ball center
NEAR_BALL_THRESHOLD_PX = 80.0
# Possession is assigned if nearest player is within this distance
POSSESSION_THRESHOLD_PX = 60.0


def _center(det: FrameDetection) -> tuple[float, float]:
    return ((det.bbox_x1 + det.bbox_x2) / 2, (det.bbox_y1 + det.bbox_y2) / 2)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def estimate_possession(
    detections: list[FrameDetection],
) -> list[PossessionEstimate]:
    """For each frame, find the player track closest to the ball.

    Returns one PossessionEstimate per frame that has a ball detection.
    """
    frames: dict[int, list[FrameDetection]] = {}
    for det in detections:
        frames.setdefault(det.frame_index, []).append(det)

    estimates: list[PossessionEstimate] = []

    for frame_index in sorted(frames.keys()):
        frame_dets = frames[frame_index]
        balls = [d for d in frame_dets if d.class_name == "ball"]
        players = [d for d in frame_dets if d.class_name == "player" and d.track_id is not None]

        if not balls:
            continue

        ball = balls[0]  # take highest-confidence ball if multiple
        ball_center = _center(ball)

        if not players:
            estimates.append(
                PossessionEstimate(
                    frame_index=frame_index,
                    timestamp_s=ball.timestamp_s,
                    possessing_track_id=None,
                    distance_px=float("inf"),
                    n_players_near_ball=0,
                )
            )
            continue

        distances = [(p, _dist(_center(p), ball_center)) for p in players]
        distances.sort(key=lambda x: x[1])
        nearest_player, nearest_dist = distances[0]

        n_near = sum(1 for _, d in distances if d <= NEAR_BALL_THRESHOLD_PX)
        possessing = nearest_player.track_id if nearest_dist <= POSSESSION_THRESHOLD_PX else None

        estimates.append(
            PossessionEstimate(
                frame_index=frame_index,
                timestamp_s=ball.timestamp_s,
                possessing_track_id=possessing,
                distance_px=nearest_dist,
                n_players_near_ball=n_near,
            )
        )

    return estimates
