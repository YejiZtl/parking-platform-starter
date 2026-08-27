import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from training_progress import update_progress


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_roi(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(data.get("points", []), dtype=np.int32)
    return points if len(points) >= 3 else None


def signature(image_path: Path, roi: np.ndarray | None) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if roi is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi], 255)
        image = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (128, 64), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)


def distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first - second)))


def select_farthest(signatures: list[np.ndarray], count: int) -> list[int]:
    if not signatures:
        return []
    selected = [0]
    while len(selected) < min(count, len(signatures)):
        best_index = None
        best_distance = -1.0
        for index, candidate in enumerate(signatures):
            if index in selected:
                continue
            nearest_distance = min(distance(candidate, signatures[chosen]) for chosen in selected)
            if nearest_distance > best_distance:
                best_index = index
                best_distance = nearest_distance
        selected.append(best_index)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select visually diverse fixed-camera frames without deleting originals.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--roi", default="parking_roi.json")
    parser.add_argument("--report", default="runs/parking_train/diverse_frames.json")
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_dir = Path(args.output)
    report_path = Path(args.report)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    roi = load_roi(Path(args.roi))
    update_progress(
        phase="candidate_selection",
        status="running",
        detail=f"Computing ROI signatures for {len(images)} fresh frames",
        completed=0,
        total=len(images),
    )
    signatures = []
    for index, image_path in enumerate(images, start=1):
        signatures.append(signature(image_path, roi))
        update_progress(
            phase="candidate_selection",
            status="running",
            detail=f"Analyzed {image_path.name}",
            completed=index,
            total=len(images),
        )

    selected_indexes = select_farthest(signatures, args.count)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for index in selected_indexes:
        source = images[index]
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        nearest_other = min(
            (distance(signatures[index], signatures[other]) for other in selected_indexes if other != index),
            default=0.0,
        )
        selected.append({"image": source.name, "nearest_selected_difference": round(nearest_other, 4)})

    report = {
        "source": str(image_dir),
        "total_frames": len(images),
        "selected_frames": len(selected),
        "output": str(output_dir),
        "selected": selected,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    update_progress(
        phase="candidate_selection",
        status="completed",
        detail=f"Selected {len(selected)} diverse frames from {len(images)} fresh captures",
        completed=len(images),
        total=len(images),
    )


if __name__ == "__main__":
    main()
