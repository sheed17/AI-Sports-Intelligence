"""Run YOLO detection over extracted frames using Ultralytics YOLOv8."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import structlog
from ultralytics import YOLO

logger = structlog.get_logger()

# COCO class IDs we care about
_COCO_PERSON = 0
_COCO_BALL = 32  # sports ball in COCO

# Map COCO label → domain label
COCO_TO_DOMAIN: dict[int, str] = {
    _COCO_PERSON: "player",
    _COCO_BALL: "ball",
}


@dataclass
class FrameDetection:
    frame_index: int
    timestamp_s: float
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    track_id: int | None = None  # filled in by tracking step


def run_detection(
    frame_paths: list[tuple[int, float, str]],  # (frame_index, timestamp_s, path)
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.3,
    batch_size: int = 16,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[FrameDetection]:
    """Run YOLOv8 inference on a list of frames.

    Returns flat list of FrameDetection objects. track_id is None until
    track_objects() runs.
    """
    model = YOLO(model_name)
    all_detections: list[FrameDetection] = []
    target_classes = list(COCO_TO_DOMAIN.keys())

    # Process in batches for efficiency
    for batch_start in range(0, len(frame_paths), batch_size):
        batch = frame_paths[batch_start : batch_start + batch_size]
        paths = [p for _, _, p in batch]

        results = model(
            paths,
            conf=confidence_threshold,
            classes=target_classes,
            verbose=False,
        )

        for (frame_index, timestamp_s, _), result in zip(batch, results):
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                domain_class = COCO_TO_DOMAIN.get(cls_id)
                if domain_class is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append(
                    FrameDetection(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        class_name=domain_class,
                        confidence=float(box.conf[0].item()),
                        bbox_x1=x1,
                        bbox_y1=y1,
                        bbox_x2=x2,
                        bbox_y2=y2,
                    )
                )

        logger.debug(
            "detection.batch_done",
            batch_start=batch_start,
            batch_size=len(batch),
            detections_so_far=len(all_detections),
        )
        if progress_callback is not None:
            progress_callback(min(batch_start + len(batch), len(frame_paths)), len(frame_paths))

    logger.info("detection.done", total_detections=len(all_detections), frames=len(frame_paths))
    return all_detections
