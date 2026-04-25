"""Extract frames from video at a configurable sample rate using OpenCV."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import structlog

logger = structlog.get_logger()


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    total_frames: int
    duration_s: float


@dataclass
class ExtractedFrame:
    frame_index: int       # original frame number in the video
    sample_index: int      # sequential index in extracted set
    timestamp_s: float
    file_path: str


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = 3,
) -> tuple[VideoMeta, list[ExtractedFrame]]:
    """Extract every `sample_rate`-th frame from `video_path` into `output_dir`.

    Returns video metadata and a list of extracted frame records.
    sample_rate=3 means every 3rd frame, so ~10 FPS from a 30 FPS source.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    meta = VideoMeta(fps=fps, width=width, height=height, total_frames=total_frames, duration_s=duration_s)
    extracted: list[ExtractedFrame] = []
    sample_index = 0
    frame_index = 0

    logger.info("extract_frames.start", video=str(video_path), fps=fps, total_frames=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_rate == 0:
            filename = f"frame_{frame_index:07d}.jpg"
            file_path = output_dir / filename
            cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            extracted.append(
                ExtractedFrame(
                    frame_index=frame_index,
                    sample_index=sample_index,
                    timestamp_s=frame_index / fps,
                    file_path=str(file_path),
                )
            )
            sample_index += 1

        frame_index += 1

    cap.release()

    # Write manifest for DVC / reproducibility
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "sample_rate": sample_rate,
                "meta": asdict(meta),
                "frames": [asdict(f) for f in extracted],
            },
            indent=2,
        )
    )

    logger.info("extract_frames.done", extracted=len(extracted), output_dir=str(output_dir))
    return meta, extracted
