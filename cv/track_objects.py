"""Assign stable track IDs across frames using Supervision + ByteTrack."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
import supervision as sv

from cv.detect_objects import FrameDetection

logger = structlog.get_logger()


@dataclass
class TrackSummary:
    track_id: int
    class_name: str
    first_seen_s: float
    last_seen_s: float
    frame_count: int


def run_tracking(
    detections: list[FrameDetection],
    frame_width: int,
    frame_height: int,
) -> tuple[list[FrameDetection], list[TrackSummary]]:
    """Run ByteTrack over ordered frame detections.

    Mutates each FrameDetection.track_id in-place and returns updated list
    plus per-track summary.
    """
    tracker = sv.ByteTrack()

    # Group detections by frame_index (already ordered from detect step)
    frames: dict[int, list[FrameDetection]] = {}
    for det in detections:
        frames.setdefault(det.frame_index, []).append(det)

    # Track summaries accumulated
    track_meta: dict[int, dict] = {}

    for frame_index in sorted(frames.keys()):
        frame_dets = frames[frame_index]
        timestamp_s = frame_dets[0].timestamp_s

        # Separate players and ball for tracking
        # ByteTrack works best per-class to avoid ID confusion
        for class_name in ("player", "ball"):
            class_dets = [d for d in frame_dets if d.class_name == class_name]
            if not class_dets:
                continue

            xyxy = np.array([[d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2] for d in class_dets])
            confs = np.array([d.confidence for d in class_dets])
            cls_ids = np.zeros(len(class_dets), dtype=int)  # all same class per iteration

            sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=cls_ids)
            tracked = tracker.update_with_detections(sv_dets)

            # Map tracked results back to FrameDetection objects
            # tracked.tracker_id aligns with input order when IDs are stable
            if tracked.tracker_id is not None:
                for i, track_id in enumerate(tracked.tracker_id):
                    if i < len(class_dets):
                        class_dets[i].track_id = int(track_id)
                        key = (class_name, int(track_id))
                        if key not in track_meta:
                            track_meta[key] = {
                                "track_id": int(track_id),
                                "class_name": class_name,
                                "first_seen_s": timestamp_s,
                                "last_seen_s": timestamp_s,
                                "frame_count": 0,
                            }
                        track_meta[key]["last_seen_s"] = timestamp_s
                        track_meta[key]["frame_count"] += 1

    summaries = [
        TrackSummary(**v)
        for v in track_meta.values()
    ]

    logger.info("tracking.done", tracks=len(summaries), detections=len(detections))
    return detections, summaries
