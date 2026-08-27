import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from parking_geometry import box_center_in_roi, deduplicate_boxes, load_roi
from training_progress import update_progress


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_classes(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def box_to_yolo(box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), float(x1)))
    x2 = max(0.0, min(float(width - 1), float(x2)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    y2 = max(0.0, min(float(height - 1), float(y2)))

    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def ownership_bounds(starts: list[int], index: int, tile_size: int, length: int):
    start = starts[index]
    left = 0.0 if index == 0 else (starts[index - 1] + tile_size + start) / 2
    right = (
        float(length)
        if index == len(starts) - 1
        else (start + tile_size + starts[index + 1]) / 2
    )
    return left, right


def predict_raw(model, source, classes, conf: float, iou: float, imgsz: int, max_det: int):
    result = model.predict(
        source=source,
        classes=classes,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], []
    return result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()


def predict_full_and_tiled(model, image, classes, args):
    boxes, scores = predict_raw(
        model, image, classes, args.conf, args.iou, args.imgsz, args.max_det
    )
    tile_raw_count = 0
    if args.tile_size <= 0:
        return boxes, scores, tile_raw_count

    height, width = image.shape[:2]
    x_starts = tile_starts(width, args.tile_size, args.tile_overlap)
    y_starts = tile_starts(height, args.tile_size, args.tile_overlap)
    for y_index, y in enumerate(y_starts):
        owner_top, owner_bottom = ownership_bounds(
            y_starts, y_index, args.tile_size, height
        )
        for x_index, x in enumerate(x_starts):
            owner_left, owner_right = ownership_bounds(
                x_starts, x_index, args.tile_size, width
            )
            tile = image[y : y + args.tile_size, x : x + args.tile_size]
            tile_boxes, tile_scores = predict_raw(
                model,
                tile,
                classes,
                args.conf,
                args.iou,
                args.tile_size,
                args.max_det,
            )
            tile_raw_count += len(tile_boxes)
            for box, score in zip(tile_boxes, tile_scores):
                center_x = x + (box[0] + box[2]) / 2
                center_y = y + (box[1] + box[3]) / 2
                if not (
                    owner_left <= center_x < owner_right
                    and owner_top <= center_y < owner_bottom
                ):
                    continue
                boxes.append([box[0] + x, box[1] + y, box[2] + x, box[3] + y])
                scores.append(score)

    if boxes:
        keep = nms(
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(scores, dtype=torch.float32),
            args.tile_merge_iou,
        ).cpu().tolist()
        boxes = [boxes[index] for index in keep]
        scores = [scores[index] for index in keep]
    return boxes, scores, tile_raw_count


def main():
    parser = argparse.ArgumentParser(description="Auto-label vehicle boxes in YOLO format.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw", help="Raw image directory.")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels", help="Output label directory.")
    parser.add_argument("--model", default="yolo26m.pt", help="Pretrained detector for pseudo labels.")
    parser.add_argument("--classes", default="2,3,5,7", help="Vehicle class IDs in the model. Use 0 for the custom one-class model.")
    parser.add_argument("--conf", type=float, default=0.08, help="Lower values produce more boxes to review.")
    parser.add_argument("--iou", type=float, default=0.35, help="Lower values remove more duplicate overlapping boxes.")
    parser.add_argument("--duplicate-iou", type=float, default=0.7, help="Explicit same-vehicle duplicate IOU.")
    parser.add_argument("--duplicate-center-ratio", type=float, default=0.2, help="Maximum normalized center distance for duplicates.")
    parser.add_argument("--duplicate-size-ratio", type=float, default=1.35, help="Maximum area ratio for duplicates.")
    parser.add_argument("--imgsz", type=int, default=1920, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=500, help="Maximum detections per image.")
    parser.add_argument("--tile-size", type=int, default=0, help="Optional overlapping tile size; 0 disables tiled inference.")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Fractional overlap between inference tiles.")
    parser.add_argument("--tile-merge-iou", type=float, default=0.35, help="NMS threshold when merging full-frame and tile detections.")
    parser.add_argument("--start", type=int, default=1, help="1-based first image index to process.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to process.")
    parser.add_argument("--roi", default="parking_roi.json", help="Optional ROI JSON. Boxes outside it are ignored.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing label files.")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    start_index = max(0, args.start - 1)
    end_index = None if args.limit is None else start_index + args.limit
    images = images[start_index:end_index]
    if not images:
        raise SystemExit("No images selected. Check --start and --limit.")

    model = YOLO(args.model)
    vehicle_classes = parse_classes(args.classes)

    update_progress(
        phase="auto_label",
        status="running",
        detail=f"Auto-labeling with {args.model}, conf={args.conf}, iou={args.iou}",
        completed=0,
        total=len(images),
    )

    for index, image_path in enumerate(images, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            print(f"Skip existing {index}/{len(images)}: {label_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        roi = load_roi(args.roi, (height, width))
        boxes, scores, tile_raw_count = predict_full_and_tiled(
            model, image, vehicle_classes, args
        )
        selected_boxes = []
        selected_scores = []
        outside_roi = 0
        for box, score in zip(boxes, scores):
            if not box_center_in_roi(box, roi):
                outside_roi += 1
                continue
            selected_boxes.append(box)
            selected_scores.append(score)

        selected_boxes, duplicates_removed = deduplicate_boxes(
            selected_boxes,
            selected_scores,
            args.duplicate_iou,
            args.duplicate_center_ratio,
            args.duplicate_size_ratio,
        )
        lines = [box_to_yolo(box, width, height) for box in selected_boxes]

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        update_progress(
            phase="auto_label",
            status="running",
            detail=f"Labeled {image_path.name}: {len(lines)} vehicles",
            completed=index,
            total=len(images),
        )
        print(
            f"Auto-labeled {index}/{len(images)}: {len(lines)} vehicles, "
            f"ignored outside ROI: {outside_roi}, removed duplicates: {duplicates_removed}, "
            f"tile raw: {tile_raw_count}"
        )

    update_progress(
        phase="auto_label",
        status="completed",
        detail=f"Auto-labeled {len(images)} images",
        completed=len(images),
        total=len(images),
    )


if __name__ == "__main__":
    main()
