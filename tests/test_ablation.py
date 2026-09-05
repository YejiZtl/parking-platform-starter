import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_ablation import (
    Variant, aggregate, filter_predictions, paired_dark_interval, score,
    split_audit, temporal_stress, validate_labels, variants,
)


class AblationTests(unittest.TestCase):
    def test_switches_retain_identical_inference_and_ground_truth(self):
        roi = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        raw = [
            {"box": [10, 10, 30, 30], "confidence": 0.9},
            {"box": [11, 10, 31, 30], "confidence": 0.8},
            {"box": [110, 10, 130, 30], "confidence": 0.7},
        ]
        gt = [{"box": [10, 10, 30, 30], "brightness": 30}]
        base = Variant("full")
        for variant, expected_predictions, expected_fp in (
            (base, 1, 0), (replace(base, roi=False), 2, 1), (replace(base, dedup=False), 2, 1),
        ):
            with self.subTest(variant=variant):
                kept, _ = filter_predictions(raw, roi, variant)
                metrics = score(gt, kept)
                self.assertEqual(metrics["gt"], 1)
                self.assertEqual(metrics["dark_tp"], 1)
                self.assertEqual(metrics["predictions"], expected_predictions)
                self.assertEqual(metrics["fp"], expected_fp)
                self.assertEqual(base.prediction_key(), variant.prediction_key())

    def test_duplicate_cannot_match_two_ground_truth_instances(self):
        gt = [{"box": [0, 0, 10, 10], "brightness": 95},
              {"box": [30, 30, 40, 40], "brightness": 200}]
        predictions = [{"box": [0, 0, 10, 10], "confidence": 0.9},
                       {"box": [0, 0, 10, 10], "confidence": 0.8}]
        result = score(gt, predictions)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["dark_tp"], 1)

    def test_empty_population_has_undefined_rates(self):
        row = {**score([], []), "raw_predictions": 0, "outside_roi_removed": 0, "duplicates_removed": 0}
        result = aggregate([row])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["dark_recall"])

    def test_training_overlap_detects_renamed_identical_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "eval.jpg"
            evaluation.write_bytes(b"same-image-bytes")
            train = root / "train"
            train.mkdir()
            (train / "renamed.jpg").write_bytes(evaluation.read_bytes())
            result = split_audit([evaluation], [train])
            self.assertEqual(result[0]["overlap"][0]["eval_file"], "eval.jpg")
            self.assertTrue(result[0]["overlap"][0]["identical_bytes"])

    def test_invalid_labels_fail_instead_of_silently_changing_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "label.txt"
            for text in ("1 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.5 0.5 0 0.1", "0 0.5 0.5"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_labels(path)

    def test_paired_bootstrap_checks_alignment(self):
        baseline = [{"image": "a", "dark_gt": 10, "dark_tp": 9}]
        same = [{"image": "a", "dark_gt": 10, "dark_tp": 9}]
        self.assertEqual(paired_dark_interval(baseline, same), [0.0, 0.0])
        with self.assertRaises(ValueError):
            paired_dark_interval(baseline, [{**same[0], "image": "b"}])

    def test_temporal_ablation_reports_both_benefit_and_release_cost(self):
        results = temporal_stress()["variants"]
        self.assertEqual(results["1"]["false_empty_steps"], 1)
        self.assertEqual(results["2"]["false_empty_steps"], 0)
        self.assertEqual(results["1"]["false_occupied_steps"], 0)
        self.assertEqual(results["2"]["false_occupied_steps"], 1)
        self.assertEqual((results["1"]["state_flips"], results["2"]["state_flips"]), (4, 2))

    def test_detection_ablations_change_one_factor(self):
        from dataclasses import asdict
        all_variants = variants()
        baseline = asdict(all_variants[0])
        for variant in all_variants[1:6]:
            changed = [key for key, value in asdict(variant).items() if key != "name" and value != baseline[key]]
            self.assertEqual(len(changed), 1, variant.name)


if __name__ == "__main__":
    unittest.main()
