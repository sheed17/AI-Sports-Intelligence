"""Generate a synthetic labeled training dataset with realistic per-class feature patterns.

Each event class has characteristic signal patterns (with Gaussian noise).
Produces enough windows to train and evaluate the LSTM with meaningful metrics.

Usage:
    python ml/generate_synthetic_data.py
    python ml/generate_synthetic_data.py --n-per-class 1500 --output-dir data/features/synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.features import (
    EVENT_CLASSES, FEATURE_DIM, WINDOW_SIZE,
    FEATURE_NAMES,
)

RNG = np.random.default_rng(42)

# ── Per-class mean vectors (one per feature dimension) ────────────────────────
# Each entry is indexed to FEATURE_NAMES order.
# Values represent the typical mean for that class mid-window.

CLASS_PROFILES: dict[str, dict] = {
    "pass": {
        "ball_x_norm":           (0.50, 0.15),   # mid-field
        "ball_y_norm":           (0.50, 0.15),
        "ball_vx_norm":          (0.08, 0.04),   # moderate velocity
        "ball_vy_norm":          (0.05, 0.03),
        "nearest_player_x":      (0.48, 0.12),
        "nearest_player_y":      (0.48, 0.12),
        "nearest_player_dist":   (0.12, 0.05),   # close contact
        "nearest_player_vx":     (0.06, 0.03),
        "nearest_player_vy":     (0.04, 0.02),
        "possession_player_enc": (0.30, 0.15),
        "possession_changed":    (0.60, 0.30),   # changes during pass
        "n_players_near_ball":   (0.27, 0.10),   # ~3 players
        "ball_in_left_third":    (0.20, 0.00),
        "ball_in_mid_third":     (0.60, 0.00),
        "ball_in_right_third":   (0.20, 0.00),
        "ball_near_goal":        (0.05, 0.00),
        "time_in_window_norm":   (0.50, 0.29),
        "frame_density":         (0.72, 0.08),
    },
    "shot": {
        "ball_x_norm":           (0.88, 0.08),   # deep in attacking third
        "ball_y_norm":           (0.50, 0.12),
        "ball_vx_norm":          (0.22, 0.06),   # fast toward goal
        "ball_vy_norm":          (0.10, 0.05),
        "nearest_player_x":      (0.85, 0.08),
        "nearest_player_y":      (0.50, 0.10),
        "nearest_player_dist":   (0.08, 0.04),   # shooter very close
        "nearest_player_vx":     (0.15, 0.05),
        "nearest_player_vy":     (0.08, 0.04),
        "possession_player_enc": (0.20, 0.10),
        "possession_changed":    (0.10, 0.15),   # rarely changes mid-shot
        "n_players_near_ball":   (0.18, 0.08),   # ~2 players
        "ball_in_left_third":    (0.05, 0.00),
        "ball_in_mid_third":     (0.05, 0.00),
        "ball_in_right_third":   (0.90, 0.00),
        "ball_near_goal":        (0.90, 0.00),   # always near goal
        "time_in_window_norm":   (0.50, 0.29),
        "frame_density":         (0.78, 0.07),
    },
    "dribble": {
        "ball_x_norm":           (0.55, 0.18),
        "ball_y_norm":           (0.45, 0.15),
        "ball_vx_norm":          (0.12, 0.04),   # consistent moderate speed
        "ball_vy_norm":          (0.04, 0.02),
        "nearest_player_x":      (0.53, 0.16),
        "nearest_player_y":      (0.43, 0.14),
        "nearest_player_dist":   (0.06, 0.03),   # very close - ball at feet
        "nearest_player_vx":     (0.11, 0.04),
        "nearest_player_vy":     (0.04, 0.02),
        "possession_player_enc": (0.35, 0.10),
        "possession_changed":    (0.05, 0.10),   # stays with same player
        "n_players_near_ball":   (0.15, 0.06),   # ~1-2 players
        "ball_in_left_third":    (0.25, 0.00),
        "ball_in_mid_third":     (0.50, 0.00),
        "ball_in_right_third":   (0.25, 0.00),
        "ball_near_goal":        (0.10, 0.00),
        "time_in_window_norm":   (0.50, 0.29),
        "frame_density":         (0.70, 0.08),
    },
    "turnover": {
        "ball_x_norm":           (0.50, 0.20),
        "ball_y_norm":           (0.50, 0.15),
        "ball_vx_norm":          (0.04, 0.03),   # ball slows down
        "ball_vy_norm":          (0.03, 0.02),
        "nearest_player_x":      (0.50, 0.18),
        "nearest_player_y":      (0.50, 0.15),
        "nearest_player_dist":   (0.10, 0.05),
        "nearest_player_vx":     (0.05, 0.03),
        "nearest_player_vy":     (0.03, 0.02),
        "possession_player_enc": (0.40, 0.20),
        "possession_changed":    (0.85, 0.20),   # hallmark: possession changes
        "n_players_near_ball":   (0.36, 0.12),   # ~4 players contesting
        "ball_in_left_third":    (0.25, 0.00),
        "ball_in_mid_third":     (0.50, 0.00),
        "ball_in_right_third":   (0.25, 0.00),
        "ball_near_goal":        (0.10, 0.00),
        "time_in_window_norm":   (0.50, 0.29),
        "frame_density":         (0.75, 0.08),
    },
    "save": {
        "ball_x_norm":           (0.10, 0.06),   # near goal (defending end)
        "ball_y_norm":           (0.50, 0.10),
        "ball_vx_norm":          (-0.15, 0.06),  # ball reverses direction
        "ball_vy_norm":          (0.05, 0.04),
        "nearest_player_x":      (0.12, 0.06),
        "nearest_player_y":      (0.50, 0.10),
        "nearest_player_dist":   (0.07, 0.03),   # keeper close to ball
        "nearest_player_vx":     (0.10, 0.05),
        "nearest_player_vy":     (0.08, 0.04),
        "possession_player_enc": (0.05, 0.03),   # usually track_id ~1 (keeper)
        "possession_changed":    (0.70, 0.25),   # possession transfers after save
        "n_players_near_ball":   (0.20, 0.08),
        "ball_in_left_third":    (0.90, 0.00),   # at defending goal end
        "ball_in_mid_third":     (0.05, 0.00),
        "ball_in_right_third":   (0.05, 0.00),
        "ball_near_goal":        (0.90, 0.00),
        "time_in_window_norm":   (0.50, 0.29),
        "frame_density":         (0.74, 0.07),
    },
    "buildup": {
        "ball_x_norm":           (0.35, 0.15),   # own half, building up
        "ball_y_norm":           (0.50, 0.18),
        "ball_vx_norm":          (0.03, 0.02),   # slow patient play
        "ball_vy_norm":          (0.02, 0.02),
        "nearest_player_x":      (0.35, 0.14),
        "nearest_player_y":      (0.50, 0.16),
        "nearest_player_dist":   (0.20, 0.08),   # more spread out
        "nearest_player_vx":     (0.03, 0.02),
        "nearest_player_vy":     (0.02, 0.02),
        "possession_player_enc": (0.25, 0.15),
        "possession_changed":    (0.20, 0.15),   # occasional short passes
        "n_players_near_ball":   (0.18, 0.08),
        "ball_in_left_third":    (0.50, 0.00),
        "ball_in_mid_third":     (0.40, 0.00),
        "ball_in_right_third":   (0.10, 0.00),
        "ball_near_goal":        (0.02, 0.00),
        "time_in_window_norm":   (0.50, 0.29),
        "frame_density":         (0.68, 0.08),
    },
}


def _generate_window(class_name: str, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Generate a single (window_size, FEATURE_DIM) window for a class."""
    profile = CLASS_PROFILES[class_name]
    window = np.zeros((window_size, FEATURE_DIM), dtype=np.float32)

    for t in range(window_size):
        time_norm = t / (window_size - 1)
        for dim, feat_name in enumerate(FEATURE_NAMES):
            if feat_name == "time_in_window_norm":
                window[t, dim] = time_norm
                continue
            mean, std = profile[feat_name]
            # Add temporal drift for realism (slight trend over window)
            drift = RNG.uniform(-0.02, 0.02)
            val = RNG.normal(mean + drift * time_norm, max(std, 0.01))
            window[t, dim] = val

        # Inject class-specific temporal events
        if class_name == "pass" and 10 <= t <= 16:
            window[t, FEATURE_NAMES.index("possession_changed")] = float(RNG.random() > 0.3)
            window[t, FEATURE_NAMES.index("ball_vx_norm")] *= 1.5

        elif class_name == "shot" and t >= window_size - 10:
            window[t, FEATURE_NAMES.index("ball_near_goal")] = 1.0
            window[t, FEATURE_NAMES.index("ball_vx_norm")] = abs(window[t, FEATURE_NAMES.index("ball_vx_norm")]) * 2

        elif class_name == "turnover" and window_size // 3 <= t <= 2 * window_size // 3:
            window[t, FEATURE_NAMES.index("possession_changed")] = float(RNG.random() > 0.2)
            window[t, FEATURE_NAMES.index("n_players_near_ball")] += 0.15

        elif class_name == "save" and t <= window_size // 2:
            window[t, FEATURE_NAMES.index("ball_near_goal")] = 1.0
            window[t, FEATURE_NAMES.index("ball_vx_norm")] = abs(window[t, FEATURE_NAMES.index("ball_vx_norm")]) * -1

        elif class_name == "dribble":
            # Possession stays constant — no changes
            window[t, FEATURE_NAMES.index("possession_changed")] = 0.0

    # Clip to valid ranges
    window = np.clip(window, -1.0, 2.0).astype(np.float32)
    # Normalize 0-1 bounded features
    for dim, name in enumerate(FEATURE_NAMES):
        if name not in ("ball_vx_norm", "ball_vy_norm", "nearest_player_vx",
                        "nearest_player_vy", "possession_changed"):
            window[:, dim] = np.clip(window[:, dim], 0.0, 1.0)

    return window


def generate_dataset(
    n_per_class: int = 1200,
    output_dir: str = "data/features/synthetic",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate balanced synthetic dataset.

    Returns:
        windows: (N, WINDOW_SIZE, FEATURE_DIM) float32
        labels:  (N,) int64
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels = []

    for class_idx, class_name in enumerate(EVENT_CLASSES):
        print(f"  Generating {n_per_class} windows for '{class_name}'...")
        for _ in range(n_per_class):
            w = _generate_window(class_name)
            all_windows.append(w)
            all_labels.append(class_idx)

    # Shuffle
    indices = np.arange(len(all_windows))
    RNG.shuffle(indices)
    windows = np.stack(all_windows)[indices].astype(np.float32)
    labels = np.array(all_labels)[indices].astype(np.int64)

    # Save
    np.save(str(output_dir / "windows.npy"), windows)
    np.save(str(output_dir / "labels.npy"), labels)

    # Write metadata
    meta = {
        "n_per_class": n_per_class,
        "total": len(windows),
        "classes": EVENT_CLASSES,
        "window_size": WINDOW_SIZE,
        "feature_dim": FEATURE_DIM,
        "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(EVENT_CLASSES)},
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDataset saved to {output_dir}/")
    print(f"  Shape: {windows.shape}")
    print(f"  Labels: {meta['class_counts']}")
    return windows, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=1200)
    parser.add_argument("--output-dir", default="data/features/synthetic")
    args = parser.parse_args()

    print(f"Generating synthetic dataset ({args.n_per_class} windows per class)...")
    generate_dataset(args.n_per_class, args.output_dir)
