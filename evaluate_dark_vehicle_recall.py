import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import cv2
import numpy as np
from ultralytics import YOLO

from auto_label_vehicles import (
    predict_full_and_tiled,
)
from parking_config import DEFAULTS
from parking_geometry import (
    area_ratio,
    box_center_in_roi,
    center_distance_ratio,
    load_roi,
    overlap_iou,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_labels(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, box_width, box_height = map(float, parts)
        boxes.append(
            [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ]
        )
    return boxes


def box_brightness(image: np.ndarray, box: list[float]) -> float:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * 0.18
    margin_y = (y2 - y1) * 0.18
    ix1 = max(0, min(width - 1, round(x1 + margin_x)))
    iy1 = max(0, min(height - 1, round(y1 + margin_y)))
    ix2 = max(ix1 + 1, min(width, round(x2 - margin_x)))
    iy2 = max(iy1 + 1, min(height, round(y2 - margin_y)))
    crop = image[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return 255.0
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(np.percentile(value, 40))


def deduplicate(detections: list[dict], iou: float, center: float, size: float):
    kept = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = detection["box"]
        duplicate = any(
            overlap_iou(box, existing["box"]) >= iou
            and center_distance_ratio(box, existing["box"]) <= center
            and area_ratio(box, existing["box"]) <= size
            for existing in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def match_predictions(ground_truth: list[dict], predictions: list[dict], threshold: float):
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            iou = overlap_iou(gt["box"], prediction["box"])
            if iou >= threshold:
                candidates.append((iou, prediction["confidence"], gt_index, prediction_index))
    matched_gt = set()
    matched_predictions = set()
    for _, _, gt_index, prediction_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
    return matched_gt, matched_predictions


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure recall separately for dark vehicles.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--classes", default="0")
    parser.add_argument("--roi", default="parking_roi.json")
    parser.add_argument("--conf", type=float, default=DEFAULTS.conf)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--dark-threshold", type=float, default=95.0)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--tile-merge-iou", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_dir = Path(args.output)
    overlay_dir = output_dir / "missed_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    roi_path = Path(args.roi) if args.roi else None
    model = YOLO(args.model)

    records = []
    image_summaries = []
    total_predictions = 0
    total_matched_predictions = 0
    for image_path in sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    ):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise SystemExit(f"Missing label: {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Could not read: {image_path}")
        height, width = image.shape[:2]
        roi = load_roi(roi_path, (height, width))
        gt_boxes = read_labels(label_path, width, height)
        ground_truth = [
            {"box": box, "brightness": box_brightness(image, box)} for box in gt_boxes
        ]

        boxes, scores, _ = predict_full_and_tiled(
            model, image, parse_classes(args.classes), args
        )
        predictions = []
        for box, confidence in zip(boxes, scores):
            if box_center_in_roi(box, roi):
                predictions.append(
                    {"box": box, "confidence": float(confidence)}
                )
        predictions = deduplicate(predictions, 0.7, 0.2, 1.35)
        matched_gt, matched_predictions = match_predictions(
            ground_truth, predictions, args.match_iou
        )
        total_predictions += len(predictions)
        total_matched_predictions += len(matched_predictions)

        canvas = image.copy()
        missed_dark = 0
        for index, item in enumerate(ground_truth):
            dark = item["brightness"] <= args.dark_threshold
            matched = index in matched_gt
            records.append(
                {
                    "image": image_path.name,
                    "index": index,
                    "brightness": round(item["brightness"], 2),
                    "dark": dark,
                    "matched": matched,
                }
            )
            if dark and not matched:
                missed_dark += 1
                x1, y1, x2, y2 = (round(value) for value in item["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"miss dark V={item['brightness']:.0f}",
                    (x1, max(24, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        if missed_dark:
            cv2.imwrite(str(overlay_dir / image_path.name), canvas)
        image_summaries.append(
            {
                "image": image_path.name,
                "ground_truth": len(ground_truth),
                "predictions": len(predictions),
                "matched": len(matched_gt),
                "missed_dark": missed_dark,
            }
        )

    dark_records = [record for record in records if record["dark"]]
    non_dark_records = [record for record in records if not record["dark"]]
    matched_all = sum(record["matched"] for record in records)
    matched_dark = sum(record["matched"] for record in dark_records)
    matched_non_dark = sum(record["matched"] for record in non_dark_records)
    summary = {
        "model": str(Path(args.model).resolve()),
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "match_iou": args.match_iou,
            "dark_threshold": args.dark_threshold,
            "tile_size": args.tile_size,
        },
        "images": len(image_summaries),
        "ground_truth": len(records),
        "predictions": total_predictions,
        "precision_at_match_iou": ratio(total_matched_predictions, total_predictions),
        "overall_recall": ratio(matched_all, len(records)),
        "dark": {
            "instances": len(dark_records),
            "matched": matched_dark,
            "missed": len(dark_records) - matched_dark,
            "recall": ratio(matched_dark, len(dark_records)),
        },
        "non_dark": {
            "instances": len(non_dark_records),
            "matched": matched_non_dark,
            "missed": len(non_dark_records) - matched_non_dark,
            "recall": ratio(matched_non_dark, len(non_dark_records)),
        },
        "per_image": image_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "instances.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys() if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
