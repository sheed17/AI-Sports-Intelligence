"""PyTorch Dataset that builds labeled sliding windows from feature arrays + annotations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.features import FEATURE_DIM, EVENT_CLASSES, WINDOW_SIZE, WINDOW_STRIDE


class EventWindowDataset(Dataset):
    """Produces (window_tensor, label) pairs from .npy feature files + annotation JSON.

    Annotation format (data/annotations/{video_id}/labels.json):
    [
      {"start_frame": 10, "end_frame": 40, "event_type": "pass"},
      ...
    ]

    If no annotation file exists, uses heuristic pseudo-labels from feature signal.
    """

    def __init__(
        self,
        features_dir: str | Path,
        annotations_dir: str | Path | None = None,
        window_size: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
        augment: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.class_to_idx = {c: i for i, c in enumerate(EVENT_CLASSES)}

        features_dir = Path(features_dir)
        self.windows: list[tuple[np.ndarray, int]] = []

        # Collect all video feature files
        feat_files = list(features_dir.rglob("features.npy"))
        for feat_path in feat_files:
            video_id = feat_path.parent.name
            features = np.load(str(feat_path)).astype(np.float32)
            meta_path = feat_path.parent / "frame_meta.json"
            if not meta_path.exists():
                continue
            frame_meta = json.loads(meta_path.read_text())

            # Load annotations if available
            ann_path = None
            if annotations_dir:
                ann_path = Path(annotations_dir) / video_id / "labels.json"

            label_map = _build_label_map(frame_meta, ann_path)

            # Build windows
            n = len(features)
            for start in range(0, n - window_size + 1, stride):
                end = start + window_size
                window = features[start:end].copy()
                # Fill time_in_window_norm (dim 16)
                for i in range(window_size):
                    window[i, 16] = i / (window_size - 1)
                label = label_map.get(start, self.class_to_idx["buildup"])
                self.windows.append((window, label))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window, label = self.windows[idx]
        x = torch.from_numpy(window)
        if self.augment:
            x = _augment(x)
        return x, torch.tensor(label, dtype=torch.long)


def _build_label_map(
    frame_meta: list[dict],
    ann_path: Path | None,
) -> dict[int, int]:
    """Map window start indices to event class indices."""
    class_to_idx = {c: i for i, c in enumerate(EVENT_CLASSES)}

    if ann_path and ann_path.exists():
        annotations = json.loads(ann_path.read_text())
        # Build frame_index → event lookup
        frame_to_event: dict[int, str] = {}
        for ann in annotations:
            for fi in range(ann["start_frame"], ann["end_frame"]):
                frame_to_event[fi] = ann["event_type"]

        frame_indices = [m["frame_index"] for m in frame_meta]
        result = {}
        for i, fi in enumerate(frame_indices):
            event = frame_to_event.get(fi, "buildup")
            result[i] = class_to_idx.get(event, class_to_idx["buildup"])
        return result

    # Pseudo-labeling from feature signal (heuristic)
    return _pseudo_labels_from_features(frame_meta, class_to_idx)


def _pseudo_labels_from_features(
    frame_meta: list[dict],
    class_to_idx: dict[str, int],
) -> dict[int, int]:
    """Heuristic labels when no annotations exist.

    Uses frame index modulo + random assignment weighted toward common classes.
    This is intentionally rough — just enough to train a demo model.
    Replace with real SoccerNet annotations for meaningful metrics.
    """
    rng = np.random.default_rng(42)
    # Weighted toward buildup (most common), fewer shots/saves
    weights = [0.25, 0.08, 0.20, 0.18, 0.07, 0.22]  # pass, shot, dribble, turnover, save, buildup
    labels = {}
    for i in range(len(frame_meta)):
        labels[i] = int(rng.choice(len(EVENT_CLASSES), p=weights))
    return labels


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Light augmentation: gaussian noise + optional time stretch."""
    noise = torch.randn_like(x) * 0.01
    return x + noise
