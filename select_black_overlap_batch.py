import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_roi(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(data.get("points", []), dtype=np.int32)
    return points if len(points) >= 3 else None


def read_boxes(path: Path, width: int, height: int) -> list[list[int]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, box_width, box_height = map(float, parts)
        boxes.append(
            [
                round((cx - box_width / 2) * width),
                round((cy - box_height / 2) * height),
                round((cx + box_width / 2) * width),
                round((cy + box_height / 2) * height),
            ]
        )
    return boxes


def dark_unboxed_candidates(
    image: np.ndarray,
    boxes: list[list[int]],
    roi: np.ndarray | None,
    scale: float = 0.25,
) -> tuple[float, list[list[int]]]:
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    search_mask = np.full((height, width), 255, dtype=np.uint8)
    if roi is not None:
        search_mask.fill(0)
        cv2.fillPoly(search_mask, [np.round(roi * scale).astype(np.int32)], 255)

    for x1, y1, x2, y2 in boxes:
        pad_x = max(3, round((x2 - x1) * scale * 0.08))
        pad_y = max(3, round((y2 - y1) * scale * 0.08))
        sx1 = max(0, round(x1 * scale) - pad_x)
        sy1 = max(0, round(y1 * scale) - pad_y)
        sx2 = min(width - 1, round(x2 * scale) + pad_x)
        sy2 = min(height - 1, round(y2 * scale) + pad_y)
        cv2.rectangle(search_mask, (sx1, sy1), (sx2, sy2), 0, -1)

    dark = ((value <= 92) & (saturation <= 125)).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, search_mask)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
    )
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    candidates = []
    component_score = 0.0
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component]
        long_side = max(box_width, box_height)
        short_side = max(1, min(box_width, box_height))
        aspect = long_side / short_side
        rectangularity = area / max(1, box_width * box_height)
        if not (45 <= area <= 1600):
            continue
        if not (9 <= short_side <= 42 and 18 <= long_side <= 80):
            continue
        if not (1.25 <= aspect <= 4.8 and rectangularity >= 0.30):
            continue

        full_box = [
            round(x / scale),
            round(y / scale),
            round((x + box_width) / scale),
            round((y + box_height) / scale),
        ]
        candidates.append(full_box)
        component_score += area * rectangularity * min(aspect, 3.0)

    label_penalty = len(boxes) * 0.8
    score = len(candidates) * 150 + component_score / 20 - label_penalty
    return score, candidates


def choose_diverse(records: list[dict], count: int) -> list[dict]:
    selected = []
    for segment in np.array_split(np.arange(len(records)), count):
        if len(segment) == 0:
            continue
        selected.append(max((records[int(index)] for index in segment), key=lambda item: item["score"]))
    return selected


def render_contact_sheets(records: list[dict], output_dir: Path) -> None:
    columns = 3
    rows = 3
    tile_width = 600
    tile_height = 365
    for sheet_index in range(math.ceil(len(records) / (columns * rows))):
        sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
        subset = records[sheet_index * columns * rows : (sheet_index + 1) * columns * rows]
        for tile_index, record in enumerate(subset):
            image = cv2.imread(str(record["source"]))
            if image is None:
                continue
            original_height, original_width = image.shape[:2]
            for x1, y1, x2, y2 in record["boxes"]:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 70), 3, cv2.LINE_AA)
            for x1, y1, x2, y2 in record["candidates"]:
                cv2.rectangle(image, (x1, y1), (x2, y2), (215, 0, 215), 4, cv2.LINE_AA)

            preview_height = tile_height - 35
            scale = min(tile_width / original_width, preview_height / original_height)
            preview = cv2.resize(
                image,
                (round(original_width * scale), round(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            row, column = divmod(tile_index, columns)
            x = column * tile_width + (tile_width - preview.shape[1]) // 2
            y = row * tile_height + 28
            sheet[y : y + preview.shape[0], x : x + preview.shape[1]] = preview
            title = f"{record['image']}  score={record['score']:.0f}  suspect={len(record['candidates'])}"
            cv2.putText(
                sheet,
                title,
                (column * tile_width + 8, row * tile_height + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(output_dir / f"contact_sheet_{sheet_index + 1:02d}.jpg"), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a diverse hard batch for missed dark vehicles.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels")
    parser.add_argument("--roi", default="parking_roi.json")
    parser.add_argument("--output", default="datasets/parking_vehicles/black_overlap_batch")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_dir = Path(args.output)
    roi = load_roi(Path(args.roi))
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if len(images) < args.count:
        raise SystemExit(f"Only {len(images)} images found; cannot select {args.count}.")

    if output_dir.exists():
        archived = output_dir.with_name(
            f"{output_dir.name}_before_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.move(str(output_dir), str(archived))
        print(f"Archived previous batch: {archived}")

    batch_image_dir = output_dir / "images"
    preview_dir = output_dir / "review"
    backup_dir = output_dir / "labels_before_manual"
    batch_image_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)

    records = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_boxes(label_path, width, height)
        score, candidates = dark_unboxed_candidates(image, boxes, roi)
        records.append(
            {
                "index": index,
                "image": image_path.name,
                "source": image_path,
                "label": label_path,
                "labels": len(boxes),
                "score": score,
                "boxes": boxes,
                "candidates": candidates,
            }
        )
        if index % 50 == 0 or index == len(images):
            print(f"Scored {index}/{len(images)}")

    selected = choose_diverse(records, args.count)
    selected.sort(key=lambda item: item["image"])
    for record in selected:
        shutil.copy2(record["source"], batch_image_dir / record["image"])
        if record["label"].exists():
            shutil.copy2(record["label"], backup_dir / record["label"].name)

    render_contact_sheets(selected, preview_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Manual correction for missed overlapping dark vehicles",
        "count": len(selected),
        "images": [
            {
                "original_index": record["index"],
                "image": record["image"],
                "existing_labels": record["labels"],
                "dark_candidates": len(record["candidates"]),
                "score": round(record["score"], 2),
            }
            for record in selected
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected {len(selected)} images into {batch_image_dir}")
    print(f"Review sheets: {preview_dir}")


if __name__ == "__main__":
    main()
