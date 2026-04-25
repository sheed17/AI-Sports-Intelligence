"""Smoke tests: verify LSTM model initializes and produces correct output shapes."""
import numpy as np
import pytest
import torch

from ml.features import FEATURE_DIM, NUM_CLASSES, WINDOW_SIZE, EVENT_CLASSES
from ml.models.lstm_classifier import LSTMEventClassifier
from ml.inference import predict_window


class TestLSTMSmoke:
    @pytest.fixture
    def model(self):
        m = LSTMEventClassifier()
        m.eval()
        return m

    def test_forward_pass_shape(self, model):
        x = torch.randn(4, WINDOW_SIZE, FEATURE_DIM)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_forward_single_sample(self, model):
        x = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_predict_proba_sums_to_one(self, model):
        x = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
        probs = model.predict_proba(x)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_predict_proba_all_positive(self, model):
        x = torch.randn(2, WINDOW_SIZE, FEATURE_DIM)
        probs = model.predict_proba(x)
        assert (probs >= 0).all()

    def test_output_is_raw_logits(self, model):
        x = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
        logits = model(x)
        # Raw logits can be negative; probabilities cannot
        # We just check the shape and that no NaN/inf values
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_num_classes_matches(self, model):
        x = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
        out = model(x)
        assert out.shape[-1] == len(EVENT_CLASSES)


class TestInferenceSmoke:
    def test_predict_window_returns_valid_class(self):
        model = LSTMEventClassifier()
        model.eval()
        window = np.random.randn(WINDOW_SIZE, FEATURE_DIM).astype(np.float32)
        event_class, confidence, top3 = predict_window(model, window)
        assert event_class in EVENT_CLASSES
        assert 0.0 <= confidence <= 1.0
        assert len(top3) == 3
        assert all(c in EVENT_CLASSES for c, _ in top3)

    def test_predict_window_wrong_shape_raises(self):
        model = LSTMEventClassifier()
        bad_window = np.zeros((10, FEATURE_DIM), dtype=np.float32)
        with pytest.raises(AssertionError):
            predict_window(model, bad_window)
