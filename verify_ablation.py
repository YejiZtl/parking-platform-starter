"""Recompute saved ablation metrics on CPU, without model weights or camera images."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from parking_geometry import load_calibration, load_parking_regions, load_roi, reference_shape_from_calibration
from run_ablation import (
    Variant, aggregate, filter_predictions, paired_dark_interval,
    score, slot_diagnostics, temporal_stress,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(directory: Path) -> None:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((directory / "raw_predictions.json").read_text(encoding="utf-8"))
    paths = summary["input_paths"]
    reference = reference_shape_from_calibration(load_calibration(paths["calibration"]))
    baseline_rows = None
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(summary["results"]), "CSV variant count mismatch")
    for result, csv_row in zip(summary["results"], csv_rows):
        variant = Variant(**result["variant"])
        evidence = raw[result["raw_prediction_source"]]
        require(json.dumps(evidence["prediction_settings"], sort_keys=True) == variant.prediction_key(),
                f"Prediction settings mismatch: {variant.name}")
        rows = []
        for frame in summary["dataset"]:
            shape = tuple(frame["shape"])
            roi = load_roi(paths["roi"], shape, reference)
            regions = load_parking_regions(paths["regions"], shape, reference)
            polygons = [np.array(item["points"], dtype=np.int32).reshape((-1, 1, 2)) for item in regions]
            kept, filters = filter_predictions(evidence["images"][frame["image"]]["detections"], roi, variant)
            rows.append({"image": frame["image"], **score(frame["ground_truth"], kept), **filters,
                         "slots": slot_diagnostics(kept, polygons)})
        require(rows == result["per_image"], f"Per-image metric mismatch: {variant.name}")
        metrics = aggregate(rows)
        metrics["mean_inference_ms"] = float(np.mean([item["speed_ms"]["inference"] for item in evidence["images"].values()]))
        require(metrics == result["metrics"], f"Aggregate mismatch: {variant.name}")
        require(csv_row["variant"] == variant.name, "CSV variant order mismatch")
        for key, value in metrics.items():
            require((csv_row[key] == "" if value is None else float(csv_row[key]) == value),
                    f"CSV mismatch: {variant.name}/{key}")
        if baseline_rows is None:
            baseline_rows = rows
        else:
            require(paired_dark_interval(baseline_rows, rows) == result["dark_recall_delta_pp_ci95"],
                    f"Bootstrap mismatch: {variant.name}")
    require(temporal_stress() == summary["temporal_stress"], "Synthetic state experiment mismatch")
    print(f"Verified {len(summary['results'])} variants, {len(summary['dataset'])} images, CSV, slot diagnostics and temporal stress.")
    print("Replay checks saved evidence consistency; it does not rerun inference or prove annotation accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/experiments/ablation_20260905"))
    verify(parser.parse_args().report)
