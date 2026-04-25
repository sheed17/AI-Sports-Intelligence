"""Standalone inference: load model artifact and predict event for a feature window."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ml.features import EVENT_CLASSES, FEATURE_DIM, WINDOW_SIZE
from ml.models.lstm_classifier import LSTMEventClassifier


def load_model(model_path: str | Path) -> LSTMEventClassifier:
    model = LSTMEventClassifier(input_size=FEATURE_DIM, num_classes=len(EVENT_CLASSES))
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_window(
    model: LSTMEventClassifier,
    window: np.ndarray,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Predict event class for a single window.

    Args:
        model: loaded LSTMEventClassifier
        window: (WINDOW_SIZE, FEATURE_DIM) float32 array

    Returns:
        (predicted_class, confidence, top3_list)
    """
    assert window.shape == (WINDOW_SIZE, FEATURE_DIM), (
        f"Expected window shape ({WINDOW_SIZE}, {FEATURE_DIM}), got {window.shape}"
    )
    tensor = torch.FloatTensor(window).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_proba(tensor).squeeze(0)

    top3 = sorted(
        [(EVENT_CLASSES[i], float(probs[i].item())) for i in range(len(EVENT_CLASSES))],
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    return top3[0][0], top3[0][1], top3


def predict_video_features(
    model: LSTMEventClassifier,
    features: np.ndarray,
    frame_meta: list[dict],
    confidence_threshold: float = 0.65,
    stride: int = 5,
) -> list[dict]:
    """Slide window over full video feature array and return confident events."""
    results = []
    n = len(features)
    for start in range(0, n - WINDOW_SIZE + 1, stride):
        end = start + WINDOW_SIZE
        window = features[start:end].copy()
        for i in range(WINDOW_SIZE):
            window[i, 16] = i / (WINDOW_SIZE - 1)
        event_type, confidence, top3 = predict_window(model, window)
        if confidence >= confidence_threshold:
            results.append({
                "event_type": event_type,
                "confidence": confidence,
                "start_time_s": frame_meta[start]["timestamp_s"],
                "end_time_s": frame_meta[min(end - 1, len(frame_meta) - 1)]["timestamp_s"],
                "start_frame": frame_meta[start]["frame_index"],
                "top3": top3,
            })
    return results
