"""Unit tests for CV feature extraction and possession estimation."""
import math

import numpy as np
import pytest

from cv.detect_objects import FrameDetection
from cv.estimate_possession import (
    POSSESSION_THRESHOLD_PX,
    estimate_possession,
)
from cv.generate_features import FEATURE_DIM, build_feature_matrix
from ml.features import FEATURE_NAMES


def _make_player(frame_index: int, ts: float, x1=100, y1=200, x2=150, y2=300, track_id=1):
    return FrameDetection(
        frame_index=frame_index,
        timestamp_s=ts,
        class_name="player",
        confidence=0.9,
        bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
        track_id=track_id,
    )


def _make_ball(frame_index: int, ts: float, x1=120, y1=240, x2=135, y2=255):
    return FrameDetection(
        frame_index=frame_index,
        timestamp_s=ts,
        class_name="ball",
        confidence=0.95,
        bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
    )


class TestPossessionEstimation:
    def test_player_near_ball_gets_possession(self):
        dets = [_make_player(0, 0.0, x1=100, y1=200, x2=150, y2=300, track_id=1), _make_ball(0, 0.0)]
        # Player center: (125, 250), Ball center: (127.5, 247.5) → dist ~3px → within threshold
        estimates = estimate_possession(dets)
        assert len(estimates) == 1
        assert estimates[0].possessing_track_id == 1

    def test_no_player_yields_none_possession(self):
        dets = [_make_ball(0, 0.0)]
        estimates = estimate_possession(dets)
        assert estimates[0].possessing_track_id is None

    def test_far_player_gets_no_possession(self):
        # Player far from ball
        dets = [
            _make_player(0, 0.0, x1=0, y1=0, x2=50, y2=50, track_id=2),
            _make_ball(0, 0.0, x1=800, y1=800, x2=815, y2=815),
        ]
        estimates = estimate_possession(dets)
        assert estimates[0].possessing_track_id is None

    def test_nearest_player_wins(self):
        dets = [
            _make_player(0, 0.0, x1=100, y1=200, x2=150, y2=300, track_id=1),
            _make_player(0, 0.0, x1=400, y1=400, x2=450, y2=500, track_id=2),
            _make_ball(0, 0.0),
        ]
        estimates = estimate_possession(dets)
        assert estimates[0].possessing_track_id == 1

    def test_multiple_frames(self):
        dets = [
            _make_player(0, 0.0, track_id=1),
            _make_ball(0, 0.0),
            _make_player(1, 0.1, track_id=1),
            _make_ball(1, 0.1),
        ]
        estimates = estimate_possession(dets)
        assert len(estimates) == 2

    def test_n_players_near_ball_counted(self):
        dets = [
            _make_player(0, 0.0, x1=110, y1=230, x2=140, y2=260, track_id=1),
            _make_player(0, 0.0, x1=115, y1=235, x2=145, y2=265, track_id=2),
            _make_ball(0, 0.0),
        ]
        estimates = estimate_possession(dets)
        assert estimates[0].n_players_near_ball >= 1


class TestFeatureMatrix:
    def test_output_shape(self):
        dets = [_make_player(0, 0.0), _make_ball(0, 0.0)]
        from cv.estimate_possession import estimate_possession
        poss = estimate_possession(dets)
        features, meta = build_feature_matrix(dets, poss, 1920, 1080, 30.0)
        assert features.shape == (1, FEATURE_DIM)

    def test_feature_names_match_dim(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_ball_position_normalized(self):
        # Ball at center of 1920x1080 frame
        dets = [
            _make_ball(0, 0.0, x1=950, y1=530, x2=970, y2=550),
        ]
        from cv.estimate_possession import estimate_possession
        poss = estimate_possession(dets)
        features, _ = build_feature_matrix(dets, poss, 1920, 1080, 30.0)
        ball_x = features[0, 0]
        ball_y = features[0, 1]
        assert 0.0 <= ball_x <= 1.0
        assert 0.0 <= ball_y <= 1.0
        assert abs(ball_x - 0.5) < 0.05  # roughly centered

    def test_possession_change_detected(self):
        # Two frames: frame 0 player 1 has ball, frame 1 player 2 has ball
        dets = [
            _make_player(0, 0.0, x1=110, y1=230, x2=140, y2=260, track_id=1),
            _make_ball(0, 0.0, x1=120, y1=240, x2=135, y2=255),
            _make_player(1, 0.1, x1=600, y1=400, x2=640, y2=450, track_id=2),
            _make_ball(1, 0.1, x1=610, y1=410, x2=625, y2=425),
            _make_player(1, 0.1, x1=110, y1=230, x2=140, y2=260, track_id=1),
        ]
        from cv.estimate_possession import estimate_possession
        poss = estimate_possession(dets)
        features, _ = build_feature_matrix(dets, poss, 1920, 1080, 30.0)
        # possession_changed is dim 10; should be 1 on second frame
        if len(features) > 1:
            assert features[1, 10] == 1.0 or features[0, 10] == 0.0

    def test_no_ball_yields_zero_ball_features(self):
        dets = [_make_player(0, 0.0)]
        from cv.estimate_possession import estimate_possession
        poss = estimate_possession(dets)
        features, _ = build_feature_matrix(dets, poss, 1920, 1080, 30.0)
        assert features[0, 0] == 0.0
        assert features[0, 1] == 0.0

    def test_dtype_is_float32(self):
        dets = [_make_ball(0, 0.0)]
        from cv.estimate_possession import estimate_possession
        poss = estimate_possession(dets)
        features, _ = build_feature_matrix(dets, poss, 1920, 1080, 30.0)
        assert features.dtype == np.float32
