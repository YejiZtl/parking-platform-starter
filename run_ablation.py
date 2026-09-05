"""Reproducible offline detection ablations; never reads .env or contacts RTSP.

Raw predictions and ground truth are saved so metrics can be independently audited.
Existing evaluation matching and production slot/state logic are reused deliberately.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from evaluate_dark_vehicle_recall import (
    box_brightness, deduplicate, match_predictions, read_labels,
)
from managed_parking import ManagedParkingManagement
from parking_geometry import (
    IMAGE_EXTS, box_center_in_roi, file_sha256, load_calibration,
    load_parking_regions, load_roi, reference_shape_from_calibration,
    validate_calibration,
)


@dataclass(frozen=True)
class Variant:
    name: str
    model: str = "releases/parking_vehicle_black_verified_v3/weights/best.pt"
    classes: tuple[int, ...] = (0,)
    conf: float = 0.12
    iou: float = 0.35
    imgsz: int = 1920
    max_det: int = 500
    roi: bool = True
    dedup: bool = True

    def prediction_key(self) -> str:
        settings = asdict(self)
        for key in ("name", "roi", "dedup"):
            settings.pop(key)
        return json.dumps(settings, sort_keys=True)


def variants() -> list[Variant]:
    baseline = Variant("full_v3")
    return [
        baseline,
        replace(baseline, name="without_roi", roi=False),
        replace(baseline, name="without_dedup", dedup=False),
        replace(baseline, name="conf_0_25", conf=0.25),
        replace(baseline, name="imgsz_1280", imgsz=1280),
        replace(baseline, name="iou_0_70", iou=0.70),
        replace(baseline, name="model_v2", model="releases/parking_vehicle_black_overlap_v2/weights/best.pt"),
        replace(baseline, name="model_v1", model="runs/parking_train/vehicle_detector_managed_v1/weights/best.pt"),
        replace(baseline, name="model_generic", model="yolo26m.pt", classes=(2, 3, 5, 7)),
    ]


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def filter_predictions(raw: list[dict], roi: np.ndarray, variant: Variant):
    inside = [item for item in raw if not variant.roi or box_center_in_roi(item["box"], roi)]
    kept = deduplicate(inside, 0.7, 0.2, 1.35) if variant.dedup else inside
    return kept, {"raw_predictions": len(raw), "outside_roi_removed": len(raw) - len(inside),
                  "duplicates_removed": len(inside) - len(kept)}


def score(ground_truth: list[dict], predictions: list[dict]) -> dict:
    matched_gt, matched_predictions = match_predictions(ground_truth, predictions, 0.5)
    dark = {index for index, item in enumerate(ground_truth) if item["brightness"] <= 95.0}
    return {
        "gt": len(ground_truth), "predictions": len(predictions), "tp": len(matched_gt),
        "fp": len(predictions) - len(matched_predictions), "fn": len(ground_truth) - len(matched_gt),
        "dark_gt": len(dark), "dark_tp": len(dark & matched_gt),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ("gt", "predictions", "tp", "fp", "fn", "dark_gt", "dark_tp",
            "raw_predictions", "outside_roi_removed", "duplicates_removed")
    totals = {key: sum(row[key] for row in rows) for key in keys}
    totals.update(
        precision=fraction(totals["tp"], totals["predictions"]),
        recall=fraction(totals["tp"], totals["gt"]),
        f1=fraction(2 * totals["tp"], 2 * totals["tp"] + totals["fp"] + totals["fn"]),
        dark_recall=fraction(totals["dark_tp"], totals["dark_gt"]),
    )
    return totals


def logic() -> ManagedParkingManagement:
    # Geometry/state methods need no model, video source or display constructor.
    parking = object.__new__(ManagedParkingManagement)
    parking.slot_overlap_threshold = 0.30
    parking.get_enclosing_box = lambda box: box
    return parking


def slot_diagnostics(predictions: list[dict], polygons: list[np.ndarray]) -> dict:
    parking = logic()
    parking.boxes = [item["box"] for item in predictions]
    parking.confs = [item["confidence"] for item in predictions]
    legacy = parking._legacy_occupied_indices(polygons)
    strict = set(parking._match_boxes_to_regions(polygons))
    return {"legacy_occupied": len(legacy), "strict_occupied": len(strict),
            "legacy_only_slot_ids": [index + 1 for index in sorted(legacy - strict)],
            "strict_only_slot_ids": [index + 1 for index in sorted(strict - legacy)]}


def temporal_stress() -> dict:
    """Synthetic mechanism check, explicitly separate from field evaluation."""
    truth = [True] * 8 + [False] * 5 + [True] * 7
    observed = truth.copy()
    observed[3] = False  # One isolated missed detection while actually occupied.
    results = {}
    for confirmations in (1, 2):
        parking = logic()
        parking.slot_states = [None]
        parking.empty_streaks = [0]
        parking.occupied_streaks = [0]
        parking.empty_confirmations = confirmations
        parking.occupied_confirmations = 1
        states = [parking._stable_state(0, value) for value in observed]
        results[str(confirmations)] = {
            "empty_confirmations": confirmations, "states": states,
            "false_empty_steps": sum(gt and not state for gt, state in zip(truth, states)),
            "false_occupied_steps": sum(not gt and state for gt, state in zip(truth, states)),
            "state_flips": sum(a != b for a, b in zip(states, states[1:])),
        }
    return {"type": "synthetic_not_field_accuracy", "interval_seconds": 5,
            "truth": truth, "observations": observed, "variants": results}


def split_audit(images: list[Path], train_dirs: list[Path]) -> list[dict]:
    eval_hashes = {file_sha256(path): path.name for path in images}
    evaluation_names = {path.name for path in images}
    result = []
    for directory in train_dirs:
        if not directory.is_dir():
            raise ValueError(f"Training directory does not exist: {directory}")
        files = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"Empty training directory: {directory}")
        overlap = []
        for path in files:
            digest = file_sha256(path)
            if path.name in evaluation_names or digest in eval_hashes:
                overlap.append({"train_file": path.name, "eval_file": eval_hashes.get(digest),
                                "identical_bytes": digest in eval_hashes})
        result.append({"directory": directory.as_posix(), "images": len(files), "overlap": overlap})
    return result


def paired_dark_interval(baseline: list[dict], candidate: list[dict]) -> list[float] | None:
    if [row["image"] for row in baseline] != [row["image"] for row in candidate]:
        raise ValueError("Paired bootstrap requires identical image order")
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(baseline), size=(5000, len(baseline)))
    denominators = np.array([row["dark_gt"] for row in baseline])[indices].sum(axis=1)
    differences = np.array([c["dark_tp"] - b["dark_tp"] for b, c in zip(baseline, candidate)])
    valid = denominators > 0
    if not valid.any():
        return None
    values = 100 * differences[indices].sum(axis=1)[valid] / denominators[valid]
    return np.percentile(values, [2.5, 97.5]).tolist()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_labels(path: Path) -> None:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if (len(values) != 5 or not all(np.isfinite(values)) or values[0] != 0
                or not all(0 <= value <= 1 for value in values[1:])
                or values[3] <= 0 or values[4] <= 0):
            raise ValueError(f"Invalid single-class YOLO label: {path}:{number}")


def load_dataset(args) -> list[dict]:
    names = [line.strip() for line in args.image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Image list must be nonempty and contain unique filenames")
    if any(Path(name).name != name or Path(name).suffix.lower() not in IMAGE_EXTS for name in names):
        raise ValueError("Image list must contain image basenames only")
    calibration = load_calibration(args.calibration)
    reference = reference_shape_from_calibration(calibration)
    frames = []
    for name in names:
        path = args.images / name
        label = args.labels / f"{path.stem}.txt"
        validate_labels(label)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        height, width = image.shape[:2]
        roi = load_roi(args.roi, (height, width), reference)
        regions = load_parking_regions(args.regions, (height, width), reference)
        validate_calibration(calibration=calibration, frame_shape=(height, width), regions=regions, roi=roi)
        gt = [{"box": box, "brightness": box_brightness(image, box)} for box in read_labels(label, width, height)]
        # Never alter the ground-truth population between ablations.
        # ROI-only annotations cannot establish full-frame precision without ROI filtering.
        frames.append({"name": name, "path": path, "image": image, "roi": roi,
                       "polygons": [np.array(item["points"], dtype=np.int32).reshape((-1, 1, 2)) for item in regions],
                       "ground_truth": gt, "image_sha256": file_sha256(path), "label_sha256": file_sha256(label),
                       "gt_outside_roi": sum(not box_center_in_roi(item["box"], roi) for item in gt)})
    return frames


def run(args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new directory to preserve evidence")
    for path in (args.roi, args.regions, args.calibration, args.image_list):
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    selected = variants()
    for variant in selected:
        if not Path(variant.model).is_file():
            raise ValueError(f"Missing model (automatic downloads disabled): {variant.model}")
    frames = load_dataset(args)
    train_dirs = args.audit_train or [
        Path("datasets/parking_vehicles/images/train"),
        Path("datasets/parking_vehicles_black_overlap_v2/images/train"),
        Path("datasets/parking_vehicles_black_verified_v3/images/train"),
    ]
    audit = split_audit([frame["path"] for frame in frames], train_dirs)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    cache = {}
    model = None
    current_model = None
    results = []
    raw_evidence = {}
    for variant in selected:
        print(f"Running {variant.name}", flush=True)
        key = variant.prediction_key()
        if key not in cache:
            if current_model != variant.model:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = YOLO(variant.model)
                current_model = variant.model
            prediction_args = dict(classes=list(variant.classes), conf=variant.conf,
                                   iou=variant.iou, imgsz=variant.imgsz, max_det=variant.max_det,
                                   device=args.device, agnostic_nms=True, half=False,
                                   augment=False, verbose=False, save=False)
            # Warm up each inference configuration. Timings exclude loading and warmup.
            model.predict(frames[0]["image"], **prediction_args)
            predictions_by_image = {}
            for frame in frames:
                prediction = model.predict(frame["image"], **prediction_args)[0]
                boxes = prediction.boxes
                raw = [] if boxes is None else [
                    {"box": box, "confidence": confidence, "class_id": int(class_id)}
                    for box, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())
                ]
                predictions_by_image[frame["name"]] = {"detections": raw, "speed_ms": prediction.speed}
            cache[key] = predictions_by_image
            raw_evidence[variant.name] = {"prediction_settings": json.loads(key), "images": predictions_by_image,
                                        "end2end": bool(getattr(model.model, "end2end", False))}
        source_name = next(name for name, item in raw_evidence.items() if json.dumps(item["prediction_settings"], sort_keys=True) == key)
        rows = []
        for frame in frames:
            raw = cache[key][frame["name"]]["detections"]
            kept, filters = filter_predictions(raw, frame["roi"], variant)
            rows.append({"image": frame["name"], **score(frame["ground_truth"], kept), **filters,
                         "slots": slot_diagnostics(kept, frame["polygons"])})
        result = {"variant": asdict(variant), "raw_prediction_source": source_name,
                  "metrics": aggregate(rows), "per_image": rows}
        result["metrics"]["mean_inference_ms"] = float(np.mean([item["speed_ms"]["inference"] for item in cache[key].values()]))
        if results:
            result["dark_recall_delta_pp_ci95"] = paired_dark_interval(results[0]["per_image"], rows)
        results.append(result)
        print(json.dumps(result["metrics"]), flush=True)
    fingerprints = {path: file_sha256(path) for path in sorted({variant.model for variant in selected})}
    source_files = ["run_ablation.py", "evaluate_dark_vehicle_recall.py", "managed_parking.py", "parking_geometry.py"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                        "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "protocol": {"match_iou": 0.5, "matching": "existing evaluator: greedy descending IoU, then confidence, one-to-one",
                     "dark_threshold": 95.0, "dark_definition": "40th percentile HSV V in central 64% of GT box width/height <= 95",
                     "gt_scope": "all supplied labels, identical across variants; outside-ROI predictions count as FP against this annotation scope",
                     "duplicate_thresholds": [0.7, 0.2, 1.35], "slot_overlap_threshold": 0.30,
                     "slot_metrics": "per-image pre-debounce diagnostics only; no manual slot truth",
                     "confidence_intervals": "paired image bootstrap, 5000 resamples, seed 0; descriptive with only 6 correlated frames",
                     "timing": "Ultralytics inference only, single measured pass per image after config warmup; cached variants share timings",
                     "seed": 0, "half": False, "augment": False, "agnostic_nms": True},
        "input_paths": {key: getattr(args, key).as_posix() for key in ("images", "labels", "image_list", "roi", "regions", "calibration")},
        "model_sha256": fingerprints,
        "source_sha256": {path: file_sha256(path) for path in source_files},
        "config_sha256": {path.as_posix(): file_sha256(path) for path in (args.roi, args.regions, args.calibration, args.image_list)},
        "dataset": [{"image": frame["name"], "image_sha256": frame["image_sha256"], "label_sha256": frame["label_sha256"],
                     "shape": list(frame["image"].shape[:2]), "ground_truth": frame["ground_truth"],
                     "gt_outside_roi": frame["gt_outside_roi"]} for frame in frames],
        "split_audit": audit,
        "limitations": ["Historical training split immutability is not proven by current files.",
                        "V3 warm-started from V2 and V1; any ancestor exposure invalidates independent-test claims.",
                        "These images served as validation for checkpoint/threshold selection, not an untouched test set.",
                        "Model replacements are checkpoint comparisons, not controlled training-component ablations.",
                        "Brightness proxy includes shadowed vehicles and does not establish paint color.",
                        "Without ROI, outside-scope detections may be real unannotated vehicles; FP is task-scope FP.",
                        "Six same-camera frames do not establish weather/night/cross-camera generalization.",
                        "YOLO end-to-end heads may bypass NMS; inspect raw end2end flags before interpreting IoU changes."],
        "results": results, "temporal_stress": temporal_stress(),
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "raw_predictions.json", raw_evidence)
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", *results[0]["metrics"].keys()])
        writer.writeheader()
        writer.writerows({"variant": item["variant"]["name"], **item["metrics"]} for item in results)
    print(f"Saved {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("datasets/parking_vehicles_black_verified_v3/images/val"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/parking_vehicles_black_verified_v3/labels/val"))
    parser.add_argument("--image-list", type=Path, default=Path("fixed_validation_images_v3.txt"))
    parser.add_argument("--roi", type=Path, default=Path("parking_roi.json"))
    parser.add_argument("--regions", type=Path, default=Path("bounding_boxes.json"))
    parser.add_argument("--calibration", type=Path, default=Path("parking_calibration.json"))
    parser.add_argument("--audit-train", type=Path, action="append", help="Repeat for every ancestor training split; current files only")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/ablation"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
