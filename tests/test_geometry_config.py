import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from parking_config import validate_runtime_args
from parking_geometry import (
    CalibrationError,
    deduplicate_boxes,
    load_parking_regions,
    load_roi,
    validate_calibration,
)


class GeometryConfigTests(unittest.TestCase):
    def test_roi_scales_from_reference_size(self):
        with tempfile.TemporaryDirectory() as directory:
            roi_path = Path(directory) / "roi.json"
            roi_path.write_text(
                json.dumps(
                    {
                        "reference_size": {"width": 100, "height": 50},
                        "points": [[0, 0], [100, 0], [100, 50], [0, 50]],
                    }
                ),
                encoding="utf-8",
            )
            roi = load_roi(roi_path, target_shape=(100, 200))
        self.assertEqual(roi.tolist(), [[0, 0], [200, 0], [200, 100], [0, 100]])

    def test_legacy_regions_scale_when_reference_shape_is_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            regions_path = Path(directory) / "spaces.json"
            regions_path.write_text(
                json.dumps([{"points": [[10, 10], [20, 10], [20, 20], [10, 20]]}]),
                encoding="utf-8",
            )
            regions = load_parking_regions(
                regions_path,
                target_shape=(200, 400),
                reference_shape=(100, 200),
            )
        self.assertEqual(regions[0]["points"], [[20, 20], [40, 20], [40, 40], [20, 40]])

    def test_duplicate_filter_keeps_highest_confidence(self):
        boxes = [[0, 0, 100, 100], [2, 2, 102, 102], [200, 200, 250, 250]]
        kept, removed = deduplicate_boxes(boxes, [0.8, 0.9, 0.7], 0.7, 0.2, 1.35)
        self.assertEqual(removed, 1)
        self.assertEqual(kept, [[2, 2, 102, 102], [200, 200, 250, 250]])

    def test_calibration_rejects_unapproved_resolution_change(self):
        calibration = {"reference": {"size": {"width": 100, "height": 50}}, "expected_spaces": 1}
        regions = [{"points": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
        with self.assertRaises(CalibrationError):
            validate_calibration(
                calibration=calibration,
                frame_shape=(60, 100),
                regions=regions,
                roi=load_roi_from_points([[0, 0], [10, 0], [10, 10]]),
            )

    def test_runtime_config_validation_rejects_invalid_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("regions.json", "roi.json", "model.pt"):
                (root / name).write_text("x", encoding="utf-8")
            args = SimpleNamespace(
                source="camera",
                regions=str(root / "regions.json"),
                roi=str(root / "roi.json"),
                model=str(root / "model.pt"),
                conf=1.5,
                iou=0.35,
                duplicate_iou=0.7,
                duplicate_center_ratio=0.2,
                slot_overlap_threshold=0.3,
                imgsz=1920,
                max_det=500,
                duplicate_size_ratio=1.35,
                empty_confirmations=2,
                occupied_confirmations=1,
                interval=5,
                reconnect_delay=2,
            )
            with self.assertRaises(ValueError):
                validate_runtime_args(args)

    def test_runtime_config_validation_rejects_missing_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("regions.json", "roi.json", "model.pt"):
                (root / name).write_text("x", encoding="utf-8")
            args = SimpleNamespace(
                source="camera",
                regions=str(root / "regions.json"),
                roi=str(root / "roi.json"),
                calibration=str(root / "missing-calibration.json"),
                model=str(root / "model.pt"),
                conf=0.12,
                iou=0.35,
                duplicate_iou=0.7,
                duplicate_center_ratio=0.2,
                slot_overlap_threshold=0.3,
                imgsz=1920,
                max_det=500,
                duplicate_size_ratio=1.35,
                empty_confirmations=2,
                occupied_confirmations=1,
                interval=5,
                reconnect_delay=2,
            )
            with self.assertRaisesRegex(ValueError, "Calibration file not found"):
                validate_runtime_args(args)


def load_roi_from_points(points):
    import numpy as np

    return np.asarray(points, dtype=np.int32)


if __name__ == "__main__":
    unittest.main()
