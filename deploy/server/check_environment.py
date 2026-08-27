from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

import torch


EXPECTED = {
    "ultralytics": "8.4.121",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "opencv-python": "5.0.0.93",
    "numpy": "2.5.2",
    "lap": "0.5.13",
    "shapely": "2.1.2",
    "python-dotenv": "1.2.3",
}


def main() -> None:
    actual = {}
    for package in EXPECTED:
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = "missing"
    report = {
        "python": platform.python_version(),
        "packages": actual,
        "torch_runtime": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    mismatches = {
        package: {"expected": expected, "actual": actual[package]}
        for package, expected in EXPECTED.items()
        if actual[package] != expected
    }
    if mismatches:
        print(json.dumps({"environment_differences": mismatches}, indent=2))

    hard_errors = {}
    missing = [package for package, value in actual.items() if value == "missing"]
    if missing:
        hard_errors["missing_packages"] = missing
    if sys.version_info[:2] != (3, 12):
        hard_errors["python"] = {"expected": "3.12.x", "actual": platform.python_version()}
    if not torch.cuda.is_available():
        hard_errors["cuda"] = {"expected": "available", "actual": "unavailable"}
    if hard_errors:
        print(json.dumps({"environment_errors": hard_errors}, indent=2), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
