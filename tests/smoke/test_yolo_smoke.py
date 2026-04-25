"""Smoke test: YOLO model loads and produces detections on a synthetic image."""
import os
import tempfile

import numpy as np
import pytest

# Skip if ultralytics is not installed (e.g., in minimal CI environment)
ultralytics = pytest.importorskip("ultralytics")


@pytest.mark.skipif(
    os.environ.get("SKIP_YOLO_SMOKE") == "1",
    reason="YOLO smoke test skipped via env var",
)
class TestYOLOSmoke:
    def test_model_loads(self):
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        assert model is not None

    def test_detection_on_blank_frame(self):
        """YOLO should not crash on a blank image (may produce zero detections)."""
        import cv2
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(f.name, blank)
            path = f.name

        try:
            results = model([path], conf=0.3, classes=[0, 32], verbose=False)
            assert results is not None
            assert len(results) == 1
        finally:
            os.unlink(path)

    def test_detection_pipeline_integration(self):
        """Test that detect_objects.run_detection returns correct types."""
        import cv2
        import tempfile
        from cv.detect_objects import run_detection, FrameDetection

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(f.name, frame)
            path = f.name

        try:
            frame_tuples = [(0, 0.0, path)]
            detections = run_detection(frame_tuples, model_name="yolov8n.pt", confidence_threshold=0.3)
            assert isinstance(detections, list)
            for d in detections:
                assert isinstance(d, FrameDetection)
                assert d.class_name in ("player", "ball")
                assert 0.0 <= d.confidence <= 1.0
        finally:
            os.unlink(path)
