import argparse
import json
from pathlib import Path

import cv2

from parking_geometry import (
    area_ratio,
    box_center_in_roi,
    center_distance_ratio,
    load_roi,
    overlap_iou,
)
from training_progress import update_progress


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_boxes(path: Path, width: int, height: int):
    boxes = []
    invalid = []
    if not path.exists():
        return boxes, ["missing_label"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            invalid.append(f"line_{line_number}_field_count")
            continue
        try:
            class_id = int(parts[0])
            cx, cy, box_width, box_height = map(float, parts[1:])
        except ValueError:
            invalid.append(f"line_{line_number}_parse")
            continue
        if class_id != 0 or not all(0.0 <= value <= 1.0 for value in (cx, cy, box_width, box_height)):
            invalid.append(f"line_{line_number}_range_or_class")
            continue
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height
        if x2 <= x1 or y2 <= y1:
            invalid.append(f"line_{line_number}_empty")
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes, invalid


def duplicate_pairs(boxes, iou_threshold: float, center_ratio_threshold: float, size_ratio_threshold: float):
    duplicates = []
    for first_index, first in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            iou = overlap_iou(first, boxes[second_index])
            center_ratio = center_distance_ratio(first, boxes[second_index])
            size_ratio = area_ratio(first, boxes[second_index])
            if iou >= iou_threshold and center_ratio <= center_ratio_threshold and size_ratio <= size_ratio_threshold:
                duplicates.append((first_index, second_index, iou))
    return duplicates


def render_overlay(image_path: Path, boxes, roi, duplicates, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    duplicate_indexes = {index for pair in duplicates for index in pair[:2]}
    if roi is not None:
        cv2.polylines(image, [roi], True, (0, 0, 255), 4)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (round(value) for value in box)
        color = (0, 0, 255) if index in duplicate_indexes else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO vehicle labels and render the most suspicious images.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels")
    parser.add_argument("--roi", default="parking_roi.json")
    parser.add_argument("--duplicate-iou", type=float, default=0.7)
    parser.add_argument("--duplicate-center-ratio", type=float, default=0.2)
    parser.add_argument("--duplicate-size-ratio", type=float, default=1.35)
    parser.add_argument("--render", type=int, default=12, help="Number of highest-risk images to render.")
    parser.add_argument("--output", default="runs/parking_train/label_audit.json")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_path = Path(args.output)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    roi = load_roi(Path(args.roi))
    records = []
    box_counts = []

    update_progress(
        phase="label_audit",
        status="running",
        detail="Checking labels, ROI filtering, and duplicate boxes",
        completed=0,
        total=len(images),
    )

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            records.append({"image": image_path.name, "errors": ["unreadable_image"], "risk": 1000})
            continue
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes, errors = read_boxes(label_path, width, height)
        duplicates = duplicate_pairs(
            boxes,
            args.duplicate_iou,
            args.duplicate_center_ratio,
            args.duplicate_size_ratio,
        )
        outside_roi = sum(not box_center_in_roi(box, roi) for box in boxes)
        tiny = sum(((box[2] - box[0]) * (box[3] - box[1])) / (width * height) < 0.00002 for box in boxes)
        risk = len(errors) * 100 + outside_roi * 20 + len(duplicates) * 10 + tiny
        box_counts.append(len(boxes))
        records.append({
            "image": image_path.name,
            "labels": len(boxes),
            "invalid": errors,
            "outside_roi": outside_roi,
            "duplicate_pairs": len(duplicates),
            "tiny_boxes": tiny,
            "risk": risk,
            "_boxes": boxes,
            "_duplicates": duplicates,
        })

        if index % 10 == 0 or index == len(images):
            update_progress(
                phase="label_audit",
                status="running",
                detail=f"Audited {image_path.name}",
                completed=index,
                total=len(images),
            )

    highest_risk = sorted(records, key=lambda item: item.get("risk", 0), reverse=True)[: args.render]
    render_dir = output_path.parent / "label_audit_overlays"
    for record in highest_risk:
        image_path = image_dir / record["image"]
        render_overlay(
            image_path,
            record.get("_boxes", []),
            roi,
            record.get("_duplicates", []),
            render_dir / record["image"],
        )

    public_records = []
    for record in records:
        public_records.append({key: value for key, value in record.items() if not key.startswith("_")})
    report = {
        "images": len(images),
        "labels": sum(box_counts),
        "minimum_labels_per_image": min(box_counts, default=0),
        "maximum_labels_per_image": max(box_counts, default=0),
        "mean_labels_per_image": round(sum(box_counts) / len(box_counts), 2) if box_counts else 0,
        "invalid_entries": sum(len(record.get("invalid", [])) for record in records),
        "outside_roi": sum(record.get("outside_roi", 0) for record in records),
        "duplicate_pairs": sum(record.get("duplicate_pairs", 0) for record in records),
        "tiny_boxes": sum(record.get("tiny_boxes", 0) for record in records),
        "render_dir": str(render_dir),
        "records": public_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))

    update_progress(
        phase="label_audit",
        status="completed",
        detail=f"Audit complete: {report['duplicate_pairs']} duplicate pairs, {report['outside_roi']} outside ROI",
        completed=len(images),
        total=len(images),
    )


if __name__ == "__main__":
    main()
