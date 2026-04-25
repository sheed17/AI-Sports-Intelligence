"""Load LSTM model and classify sliding windows of features into events."""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import structlog
import torch
from sqlalchemy.orm import Session

from backend.config import get_settings
from ml.features import FEATURE_DIM, EVENT_CLASSES, WINDOW_SIZE, WINDOW_STRIDE

logger = structlog.get_logger()


def _load_model(model_path: str | Path | None = None):
    """Load LSTM classifier. Falls back to untrained model if no artifact found."""
    from ml.models.lstm_classifier import LSTMEventClassifier

    settings = get_settings()
    model = LSTMEventClassifier(input_size=FEATURE_DIM, num_classes=len(EVENT_CLASSES))
    model.eval()

    if model_path is None:
        candidate = settings.models_dir / "best_model.pth"
        if candidate.exists():
            model_path = candidate

    if model_path and Path(model_path).exists():
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info("event_service.model_loaded", path=str(model_path))
    else:
        logger.warning("event_service.no_model_artifact_using_random_weights")

    return model


def _nms_merge_windows(
    windows: list[dict], iou_threshold_s: float = 2.0
) -> list[dict]:
    """Merge overlapping event windows by keeping the highest-confidence one."""
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w["confidence"], reverse=True)
    kept = []
    for w in windows:
        overlap = any(
            w["event_type"] == k["event_type"]
            and w["start_time_s"] < k["end_time_s"]
            and w["end_time_s"] > k["start_time_s"]
            for k in kept
        )
        if not overlap:
            kept.append(w)
    return sorted(kept, key=lambda w: w["start_time_s"])


def classify_and_store_events(
    session: Session,
    video_id: str,
    features: np.ndarray,
    frame_meta: list[dict],
    fps: float,
    confidence_threshold: float = 0.65,
    model_path: str | None = None,
) -> int:
    """Slide a window over features, classify each, store confident events.

    Returns count of events stored.
    """
    model = _load_model(model_path)
    n_frames = len(features)
    candidates: list[dict] = []

    with torch.no_grad():
        for start in range(0, n_frames - WINDOW_SIZE + 1, WINDOW_STRIDE):
            end = start + WINDOW_SIZE
            window = features[start:end].copy()

            # Fill time_in_window_norm (dim 16)
            for i in range(WINDOW_SIZE):
                window[i, 16] = i / (WINDOW_SIZE - 1)

            tensor = torch.FloatTensor(window).unsqueeze(0)  # (1, seq, feat)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top_conf, top_idx = probs.max(dim=0)
            conf = top_conf.item()

            if conf < confidence_threshold:
                continue

            event_class = EVENT_CLASSES[top_idx.item()]
            start_ts = frame_meta[start]["timestamp_s"]
            end_ts = frame_meta[min(end - 1, len(frame_meta) - 1)]["timestamp_s"]

            # Build top-3 class breakdown for metadata
            top3 = sorted(
                [(EVENT_CLASSES[i], float(probs[i].item())) for i in range(len(EVENT_CLASSES))],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            candidates.append(
                {
                    "event_type": event_class,
                    "confidence": conf,
                    "start_time_s": start_ts,
                    "end_time_s": end_ts,
                    "start_frame": frame_meta[start]["frame_index"],
                    "end_frame": frame_meta[min(end - 1, len(frame_meta) - 1)]["frame_index"],
                    "top3_classes": top3,
                }
            )

    merged = _nms_merge_windows(candidates)
    from backend.db.models import EventPrediction

    event_objects = []
    for ev in merged:
        event_objects.append(
            EventPrediction(
                id=str(uuid.uuid4()),
                video_id=video_id,
                event_type=ev["event_type"],
                confidence=ev["confidence"],
                start_time_s=ev["start_time_s"],
                end_time_s=ev["end_time_s"],
                start_frame=ev["start_frame"],
                end_frame=ev["end_frame"],
                metadata_={"top3": ev["top3_classes"]},
            )
        )

    session.bulk_save_objects(event_objects)
    session.commit()
    logger.info("event_service.events_stored", count=len(event_objects), video_id=video_id)
    return len(event_objects)
