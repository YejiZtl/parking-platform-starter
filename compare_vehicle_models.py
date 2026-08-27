import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import cv2
import numpy as np
from ultralytics import YOLO

from parking_config import DEFAULTS
from parking_geometry import area_ratio, center_distance_ratio, load_roi, overlap_iou
from training_progress import update_progress


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_classes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def center_in_roi(box: list[float], roi: np.ndarray | None) -> bool:
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return cv2.pointPolygonTest(roi, center, False) >= 0


def deduplicate(detections: list[dict], iou: float, center_ratio: float, size_ratio: float):
    kept: list[dict] = []
    removed = 0
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = detection["box"]
        duplicate = any(
            overlap_iou(box, existing["box"]) >= iou
            and center_distance_ratio(box, existing["box"]) <= center_ratio
            and area_ratio(box, existing["box"]) <= size_ratio
            for existing in kept
        )
        if duplicate:
            removed += 1
        else:
            kept.append(detection)
    return kept, removed


def predict(
    model: YOLO,
    image_path: Path,
    classes: list[int],
    roi: np.ndarray | None,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    duplicate_iou: float,
    duplicate_center_ratio: float,
    duplicate_size_ratio: float,
):
    result = model.predict(
        source=str(image_path),
        classes=classes,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )[0]

    raw: list[dict] = []
    if result.boxes is not None:
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            raw.append(
                {
                    "box": [round(float(value), 2) for value in box],
                    "confidence": round(float(score), 5),
                    "class_id": int(class_id),
                }
            )

    inside = [detection for detection in raw if center_in_roi(detection["box"], roi)]
    kept, duplicates_removed = deduplicate(
        inside,
        duplicate_iou,
        duplicate_center_ratio,
        duplicate_size_ratio,
    )
    return {
        "raw_count": len(raw),
        "outside_roi": len(raw) - len(inside),
        "duplicates_removed": duplicates_removed,
        "count": len(kept),
        "mean_confidence": (
            round(sum(item["confidence"] for item in kept) / len(kept), 5)
            if kept
            else 0.0
        ),
        "detections": kept,
    }


def draw(image: np.ndarray, result: dict, roi: np.ndarray | None, title: str) -> np.ndarray:
    canvas = image.copy()
    if roi is not None:
        cv2.polylines(canvas, [roi], True, (0, 0, 255), 4, cv2.LINE_AA)

    for detection in result["detections"]:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
        confidence = detection["confidence"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 220, 30), 2, cv2.LINE_AA)
        label = f"{confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_y = max(text_height + 3, y1)
        cv2.rectangle(
            canvas,
            (x1, label_y - text_height - 3),
            (x1 + text_width + 4, label_y + 2),
            (30, 220, 30),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    header = (
        f"{title} | count={result['count']} | outside={result['outside_roi']} "
        f"| duplicates={result['duplicates_removed']}"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1050), 54), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        header,
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def match_detections(baseline: list[dict], candidate: list[dict]):
    unmatched_baseline = set(range(len(baseline)))
    matches: list[tuple[int, int]] = []
    candidate_only: list[int] = []

    for candidate_index, candidate_detection in enumerate(candidate):
        candidate_box = candidate_detection["box"]
        eligible = []
        for baseline_index in unmatched_baseline:
            baseline_box = baseline[baseline_index]["box"]
            iou = overlap_iou(candidate_box, baseline_box)
            close_center = (
                center_distance_ratio(candidate_box, baseline_box) <= 0.3
                and area_ratio(candidate_box, baseline_box) <= 2.0
            )
            if iou >= 0.3 or close_center:
                eligible.append((iou, baseline_index))

        if not eligible:
            candidate_only.append(candidate_index)
            continue

        _, baseline_index = max(eligible)
        unmatched_baseline.remove(baseline_index)
        matches.append((baseline_index, candidate_index))

    return {
        "matches": matches,
        "baseline_only": sorted(unmatched_baseline),
        "candidate_only": candidate_only,
    }


def draw_differences(
    image: np.ndarray,
    baseline: list[dict],
    candidate: list[dict],
    matching: dict,
    roi: np.ndarray | None,
) -> np.ndarray:
    canvas = image.copy()
    if roi is not None:
        cv2.polylines(canvas, [roi], True, (0, 0, 255), 4, cv2.LINE_AA)

    groups = (
        (matching["baseline_only"], baseline, (0, 165, 255), "B"),
        (matching["candidate_only"], candidate, (255, 255, 0), "C"),
    )
    for indices, detections, color, prefix in groups:
        for index in indices:
            detection = detections[index]
            x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 5, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{prefix}:{detection['confidence']:.2f}",
                (x1, max(24, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

    header = (
        f"DIFFERENCES | baseline-only={len(matching['baseline_only'])} "
        f"| candidate-only={len(matching['candidate_only'])}"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1050), 54), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        header,
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and custom vehicle detectors.")
    parser.add_argument("--images", required=True, help="Directory containing external test images.")
    parser.add_argument("--baseline", default="yolo26m.pt", help="Baseline model path.")
    parser.add_argument("--candidate", required=True, help="Candidate custom model path.")
    parser.add_argument("--baseline-classes", default="2,3,5,7")
    parser.add_argument("--candidate-classes", default="0")
    parser.add_argument("--roi", default="parking_roi.json")
    parser.add_argument("--output", default="runs/parking_train/external_comparison")
    parser.add_argument("--conf", type=float, default=DEFAULTS.conf)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--duplicate-iou", type=float, default=0.7)
    parser.add_argument("--duplicate-center-ratio", type=float, default=0.2)
    parser.add_argument("--duplicate-size-ratio", type=float, default=1.35)
    args = parser.parse_args()

    image_dir = Path(args.images)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    output_dir = Path(args.output)
    baseline_dir = output_dir / "baseline"
    candidate_dir = output_dir / "candidate"
    comparison_dir = output_dir / "side_by_side"
    difference_dir = output_dir / "differences"
    for path in (baseline_dir, candidate_dir, comparison_dir, difference_dir):
        path.mkdir(parents=True, exist_ok=True)

    baseline = YOLO(args.baseline)
    candidate = YOLO(args.candidate)
    baseline_classes = parse_classes(args.baseline_classes)
    candidate_classes = parse_classes(args.candidate_classes)

    report = {
        "settings": {
            "images": str(image_dir.resolve()),
            "baseline": str(Path(args.baseline).resolve()),
            "candidate": str(Path(args.candidate).resolve()),
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "roi": str(Path(args.roi).resolve()),
        },
        "images": [],
    }
    update_progress(
        phase="external_validation",
        status="running",
        detail="Comparing baseline and trained models on fresh RTSP frames",
        completed=0,
        total=len(images),
    )

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        roi = load_roi(args.roi, image.shape[:2])

        common = {
            "image_path": image_path,
            "roi": roi,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "duplicate_iou": args.duplicate_iou,
            "duplicate_center_ratio": args.duplicate_center_ratio,
            "duplicate_size_ratio": args.duplicate_size_ratio,
        }
        baseline_result = predict(baseline, classes=baseline_classes, **common)
        candidate_result = predict(candidate, classes=candidate_classes, **common)
        matching = match_detections(
            baseline_result["detections"], candidate_result["detections"]
        )
        baseline_image = draw(image, baseline_result, roi, "BASELINE yolo26m")
        candidate_image = draw(image, candidate_result, roi, "TRAINED best.pt")
        difference_image = draw_differences(
            image,
            baseline_result["detections"],
            candidate_result["detections"],
            matching,
            roi,
        )

        cv2.imwrite(str(baseline_dir / image_path.name), baseline_image)
        cv2.imwrite(str(candidate_dir / image_path.name), candidate_image)
        cv2.imwrite(
            str(comparison_dir / image_path.name),
            np.concatenate((baseline_image, candidate_image), axis=1),
        )
        cv2.imwrite(str(difference_dir / image_path.name), difference_image)

        report["images"].append(
            {
                "image": image_path.name,
                "baseline": baseline_result,
                "candidate": candidate_result,
                "count_difference": candidate_result["count"] - baseline_result["count"],
                "matching": matching,
            }
        )
        update_progress(
            phase="external_validation",
            status="running",
            detail=f"Compared {image_path.name}",
            completed=index,
            total=len(images),
        )
        print(
            f"{index}/{len(images)} {image_path.name}: "
            f"baseline={baseline_result['count']}, candidate={candidate_result['count']}"
        )

    report["summary"] = {
        "image_count": len(images),
        "baseline_mean_count": round(
            sum(item["baseline"]["count"] for item in report["images"]) / len(images), 2
        ),
        "candidate_mean_count": round(
            sum(item["candidate"]["count"] for item in report["images"]) / len(images), 2
        ),
        "baseline_outside_roi": sum(
            item["baseline"]["outside_roi"] for item in report["images"]
        ),
        "candidate_outside_roi": sum(
            item["candidate"]["outside_roi"] for item in report["images"]
        ),
        "baseline_duplicates_removed": sum(
            item["baseline"]["duplicates_removed"] for item in report["images"]
        ),
        "candidate_duplicates_removed": sum(
            item["candidate"]["duplicates_removed"] for item in report["images"]
        ),
        "baseline_only": sum(
            len(item["matching"]["baseline_only"]) for item in report["images"]
        ),
        "candidate_only": sum(
            len(item["matching"]["candidate_only"]) for item in report["images"]
        ),
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_progress(
        phase="external_validation",
        status="completed",
        detail=f"External comparison saved to {report_path}",
        completed=len(images),
        total=len(images),
        latest_model=Path(args.candidate).resolve(),
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
