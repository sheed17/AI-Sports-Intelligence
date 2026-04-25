"""Convert per-frame detections + possession estimates into an (N, 18) feature array.

Feature vector layout (18 dims) — defined in ml/features.py as constants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from cv.detect_objects import FrameDetection
from cv.estimate_possession import PossessionEstimate

logger = structlog.get_logger()

# Canonical feature dimension — must match ml/features.py
FEATURE_DIM = 18


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def build_feature_matrix(
    detections: list[FrameDetection],
    possession_estimates: list[PossessionEstimate],
    video_width: int,
    video_height: int,
    fps: float,
) -> tuple[np.ndarray, list[dict]]:
    """Build (N_frames, FEATURE_DIM) feature matrix from detections.

    Returns:
        features: float32 array of shape (N, 18)
        frame_meta: list of {frame_index, timestamp_s} dicts — one per row
    """
    # Index detections and possession by frame
    dets_by_frame: dict[int, list[FrameDetection]] = {}
    for d in detections:
        dets_by_frame.setdefault(d.frame_index, []).append(d)

    poss_by_frame: dict[int, PossessionEstimate] = {p.frame_index: p for p in possession_estimates}

    all_frames = sorted(dets_by_frame.keys())
    rows: list[np.ndarray] = []
    meta: list[dict] = []

    prev_ball_center: tuple[float, float] | None = None
    prev_possessing_track: int | None = None

    for frame_index in all_frames:
        frame_dets = dets_by_frame[frame_index]
        poss = poss_by_frame.get(frame_index)
        timestamp_s = frame_dets[0].timestamp_s

        balls = [d for d in frame_dets if d.class_name == "ball"]
        players = [d for d in frame_dets if d.class_name == "player"]

        # Ball features
        if balls:
            ball = balls[0]
            bx = _safe_div((ball.bbox_x1 + ball.bbox_x2) / 2, video_width)
            by = _safe_div((ball.bbox_y1 + ball.bbox_y2) / 2, video_height)
            cur_center = (bx * video_width, by * video_height)
            if prev_ball_center:
                bvx = _safe_div(cur_center[0] - prev_ball_center[0], video_width)
                bvy = _safe_div(cur_center[1] - prev_ball_center[1], video_height)
            else:
                bvx, bvy = 0.0, 0.0
            prev_ball_center = cur_center
        else:
            bx, by, bvx, bvy = 0.0, 0.0, 0.0, 0.0

        # Nearest player features
        if poss and players:
            # Find nearest player det
            nearest_p = None
            min_dist = float("inf")
            for p in players:
                if p.track_id == poss.possessing_track_id:
                    nearest_p = p
                    min_dist = poss.distance_px
                    break
            if nearest_p is None and players:
                nearest_p = players[0]
                min_dist = poss.distance_px if poss else 0.0

            if nearest_p:
                npx = _safe_div((nearest_p.bbox_x1 + nearest_p.bbox_x2) / 2, video_width)
                npy = _safe_div((nearest_p.bbox_y1 + nearest_p.bbox_y2) / 2, video_height)
                npd = _safe_div(min_dist, max(video_width, video_height))
                # velocity: approximate as bbox center movement (no prev frame linkage here)
                npvx, npvy = 0.0, 0.0
            else:
                npx, npy, npd, npvx, npvy = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            npx, npy, npd, npvx, npvy = 0.0, 0.0, 0.0, 0.0, 0.0

        # Possession features
        possessing_track = poss.possessing_track_id if poss else None
        possession_changed = 1.0 if (
            possessing_track != prev_possessing_track
            and possessing_track is not None
            and prev_possessing_track is not None
        ) else 0.0
        prev_possessing_track = possessing_track
        n_near_norm = _safe_div(poss.n_players_near_ball if poss else 0, 11)

        # Field zone features (based on normalized ball_x)
        ball_in_left = 1.0 if bx < 0.33 else 0.0
        ball_in_mid = 1.0 if 0.33 <= bx < 0.67 else 0.0
        ball_in_right = 1.0 if bx >= 0.67 else 0.0
        ball_near_goal = 1.0 if (bx < 0.1 or bx > 0.9) and (0.3 < by < 0.7) else 0.0

        # Possession track encoded (ordinal, bounded)
        poss_enc = _safe_div(min(possessing_track or 0, 50), 50)

        # Frame density: avg detection confidence in frame
        all_confs = [d.confidence for d in frame_dets]
        frame_density = float(np.mean(all_confs)) if all_confs else 0.0

        row = np.array([
            bx, by,            # 0,1: ball position
            bvx, bvy,          # 2,3: ball velocity
            npx, npy,          # 4,5: nearest player position
            npd,               # 6: nearest player distance (norm)
            npvx, npvy,        # 7,8: nearest player velocity
            poss_enc,          # 9: possession player encoded
            possession_changed, # 10
            n_near_norm,       # 11: players near ball (norm)
            ball_in_left,      # 12
            ball_in_mid,       # 13
            ball_in_right,     # 14
            ball_near_goal,    # 15
            0.0,               # 16: time_in_window_norm (filled during windowing)
            frame_density,     # 17
        ], dtype=np.float32)

        rows.append(row)
        meta.append({"frame_index": frame_index, "timestamp_s": timestamp_s})

    features = np.stack(rows) if rows else np.empty((0, FEATURE_DIM), dtype=np.float32)
    logger.info("generate_features.done", shape=features.shape)
    return features, meta


def save_features(
    features: np.ndarray,
    meta: list[dict],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist feature matrix and metadata to disk for DVC tracking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_path = output_dir / "features.npy"
    meta_path = output_dir / "frame_meta.json"
    np.save(str(feat_path), features)
    meta_path.write_text(json.dumps(meta, indent=2))
    return feat_path, meta_path


def load_features(features_dir: str | Path) -> tuple[np.ndarray, list[dict]]:
    features_dir = Path(features_dir)
    features = np.load(str(features_dir / "features.npy"))
    meta = json.loads((features_dir / "frame_meta.json").read_text())
    return features, meta
