import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_roi(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    points = np.array(data["points"], dtype=np.int32)
    if len(points) < 3:
        raise SystemExit("ROI must contain at least 3 points.")
    return points


def yolo_center_in_roi(line: str, roi, width: int, height: int) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False
    _, cx, cy, bw, bh = map(float, parts)
    x = cx * width
    y = cy * height
    return cv2.pointPolygonTest(roi, (x, y), False) >= 0


def main():
    parser = argparse.ArgumentParser(description="Remove YOLO labels whose centers are outside the ROI.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw", help="Raw image directory.")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels", help="YOLO label directory.")
    parser.add_argument("--roi", default="parking_roi.json", help="ROI JSON file.")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    roi = load_roi(Path(args.roi))

    total_removed = 0
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        lines = label_path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if yolo_center_in_roi(line, roi, width, height)]
        removed = len(lines) - len(kept)
        total_removed += removed
        if removed:
            label_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            print(f"{image_path.name}: removed {removed}")

    print(f"Done. Removed {total_removed} boxes outside ROI.")


if __name__ == "__main__":
    main()
