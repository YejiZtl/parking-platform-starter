from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class CalibrationError(ValueError):
    pass


def shape_from_image(path: str | Path | None) -> tuple[int, int] | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return height, width


def normalize_shape(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
        if width and height:
            return int(height), int(width)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first, second = int(value[0]), int(value[1])
        return first, second
    return None


def scale_points(
    points: Any,
    *,
    reference_shape: tuple[int, int] | None,
    target_shape: tuple[int, int] | None,
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationError("Points must be a list of [x, y] pairs.")
    if len(array) < 3:
        raise CalibrationError("A polygon must contain at least 3 points.")
    if reference_shape and target_shape and reference_shape != target_shape:
        reference_height, reference_width = reference_shape
        target_height, target_width = target_shape
        array[:, 0] *= target_width / reference_width
        array[:, 1] *= target_height / reference_height
    return np.round(array).astype(np.int32)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reference_shape_from_item(item: dict[str, Any], base_dir: Path) -> tuple[int, int] | None:
    for key in ("reference_shape", "reference_size", "image_shape", "image_size"):
        shape = normalize_shape(item.get(key))
        if shape:
            return shape

    width = item.get("reference_width") or item.get("width")
    height = item.get("reference_height") or item.get("height")
    if width and height:
        return int(height), int(width)

    image_name = item.get("image") or item.get("reference_image")
    if image_name:
        return shape_from_image(base_dir / str(image_name))
    return None


def load_roi(
    path: str | Path | None,
    target_shape: tuple[int, int] | None = None,
    reference_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    if not path:
        return None
    roi_path = Path(path)
    if not roi_path.exists():
        return None
    data = load_json(roi_path)
    if not isinstance(data, dict):
        raise CalibrationError(f"ROI JSON must be an object: {roi_path}")
    shape = reference_shape or reference_shape_from_item(data, roi_path.parent)
    return scale_points(data.get("points", []), reference_shape=shape, target_shape=target_shape)


def load_parking_regions(
    path: str | Path,
    target_shape: tuple[int, int] | None = None,
    reference_shape: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    json_path = Path(path)
    data = load_json(json_path)
    if isinstance(data, dict):
        regions = data.get("regions") or data.get("spaces") or data.get("parking_spaces")
        shape = reference_shape or reference_shape_from_item(data, json_path.parent)
    else:
        regions = data
        shape = reference_shape

    if not isinstance(regions, list):
        raise CalibrationError(f"Parking region JSON must contain a list: {json_path}")

    loaded: list[dict[str, Any]] = []
    for index, region in enumerate(regions, start=1):
        if not isinstance(region, dict):
            raise CalibrationError(f"Parking space {index} must be an object.")
        polygon = scale_points(
            region.get("points", []),
            reference_shape=shape,
            target_shape=target_shape,
        )
        copied = dict(region)
        copied["points"] = polygon.reshape((-1, 2)).astype(int).tolist()
        loaded.append(copied)
    return loaded


def load_calibration(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    calibration_path = Path(path)
    if not calibration_path.exists():
        return {}
    data = load_json(calibration_path)
    if not isinstance(data, dict):
        raise CalibrationError(f"Calibration file must be a JSON object: {calibration_path}")
    return data


def reference_shape_from_calibration(calibration: dict[str, Any]) -> tuple[int, int] | None:
    reference = calibration.get("reference") if isinstance(calibration.get("reference"), dict) else {}
    return (
        normalize_shape(reference.get("shape"))
        or normalize_shape(reference.get("size"))
        or normalize_shape(calibration.get("reference_shape"))
        or normalize_shape(calibration.get("reference_size"))
    )


def validate_calibration(
    *,
    calibration: dict[str, Any],
    frame_shape: tuple[int, int],
    regions: list[dict[str, Any]],
    roi: np.ndarray | None,
    allow_scale: bool = False,
) -> None:
    reference_shape = reference_shape_from_calibration(calibration)
    expected_spaces = calibration.get("expected_spaces")
    if expected_spaces is not None and len(regions) != int(expected_spaces):
        raise CalibrationError(
            f"Parking space count mismatch: expected {expected_spaces}, got {len(regions)}."
        )
    if roi is None or len(roi) < 3:
        raise CalibrationError("ROI is missing or invalid.")
    if reference_shape and reference_shape != frame_shape and not allow_scale:
        raise CalibrationError(
            "Camera frame resolution does not match calibration: "
            f"expected {reference_shape[1]}x{reference_shape[0]}, "
            f"got {frame_shape[1]}x{frame_shape[0]}."
        )


def box_center_in_roi(box: list[float] | tuple[float, ...], roi: np.ndarray | None) -> bool:
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    center = ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)
    return cv2.pointPolygonTest(roi, center, False) >= 0


def overlap_iou(first: list[float] | tuple[float, ...], second: list[float] | tuple[float, ...]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_ratio(first: list[float] | tuple[float, ...], second: list[float] | tuple[float, ...]) -> float:
    first_center = ((float(first[0]) + float(first[2])) / 2, (float(first[1]) + float(first[3])) / 2)
    second_center = ((float(second[0]) + float(second[2])) / 2, (float(second[1]) + float(second[3])) / 2)
    average_width = ((float(first[2]) - float(first[0])) + (float(second[2]) - float(second[0]))) / 2
    average_height = ((float(first[3]) - float(first[1])) + (float(second[3]) - float(second[1]))) / 2
    dx = (first_center[0] - second_center[0]) / max(average_width, 1.0)
    dy = (first_center[1] - second_center[1]) / max(average_height, 1.0)
    return float(np.hypot(dx, dy))


def area_ratio(first: list[float] | tuple[float, ...], second: list[float] | tuple[float, ...]) -> float:
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    return max(first_area, second_area) / max(min(first_area, second_area), 1.0)


def deduplicate_boxes(
    boxes: list[list[float]],
    scores: list[float],
    iou_threshold: float,
    center_ratio_threshold: float,
    size_ratio_threshold: float,
) -> tuple[list[list[float]], int]:
    kept: list[tuple[list[float], float]] = []
    removed = 0
    for index in sorted(range(len(boxes)), key=lambda item: scores[item], reverse=True):
        candidate = boxes[index]
        duplicate = any(
            overlap_iou(candidate, box) >= iou_threshold
            and center_distance_ratio(candidate, box) <= center_ratio_threshold
            and area_ratio(candidate, box) <= size_ratio_threshold
            for box, _score in kept
        )
        if duplicate:
            removed += 1
        else:
            kept.append((candidate, scores[index]))
    return [box for box, _score in kept], removed
