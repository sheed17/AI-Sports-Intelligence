"""Canonical feature schema — single source of truth for feature dimension and names.

Any change here must be reflected in cv/generate_features.py and the LSTM input.
"""

FEATURE_DIM = 18

FEATURE_NAMES = [
    "ball_x_norm",           # 0  — ball center x, normalized 0..1
    "ball_y_norm",           # 1  — ball center y, normalized 0..1
    "ball_vx_norm",          # 2  — ball velocity x, normalized by frame width
    "ball_vy_norm",          # 3  — ball velocity y, normalized by frame height
    "nearest_player_x",      # 4  — nearest player center x, norm
    "nearest_player_y",      # 5  — nearest player center y, norm
    "nearest_player_dist",   # 6  — nearest player distance to ball, norm
    "nearest_player_vx",     # 7  — nearest player velocity x
    "nearest_player_vy",     # 8  — nearest player velocity y
    "possession_player_enc", # 9  — possessing track_id, ordinal encoded 0..1
    "possession_changed",    # 10 — binary: possession changed this frame
    "n_players_near_ball",   # 11 — count of players within threshold, norm by 11
    "ball_in_left_third",    # 12 — binary zone feature
    "ball_in_mid_third",     # 13 — binary zone feature
    "ball_in_right_third",   # 14 — binary zone feature
    "ball_near_goal",        # 15 — binary: ball in goal zone
    "time_in_window_norm",   # 16 — position within current window, 0..1
    "frame_density",         # 17 — mean detection confidence in frame
]

# Event class labels — index must match LSTM output neuron order
EVENT_CLASSES = [
    "pass",       # 0
    "shot",       # 1
    "dribble",    # 2
    "turnover",   # 3
    "save",       # 4
    "buildup",    # 5
]

NUM_CLASSES = len(EVENT_CLASSES)

# Sliding window parameters
WINDOW_SIZE = 30    # frames per window (~1 second at 30fps, ~3s at 10fps)
WINDOW_STRIDE = 5   # stride between windows

assert len(FEATURE_NAMES) == FEATURE_DIM, "FEATURE_NAMES must match FEATURE_DIM"
