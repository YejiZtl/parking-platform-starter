import argparse
import json
import random
import shutil
from pathlib import Path

from parking_geometry import file_sha256
from training_progress import update_progress


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def copy_pair(image_path: Path, label_path: Path, image_out: Path, label_out: Path):
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out)
    shutil.copy2(label_path, label_out)


def write_data_yaml(dataset: Path) -> None:
    (dataset / "data.yaml").write_text(
        "\n".join(
            (
                "path: .",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: vehicle",
                "",
            )
        ),
        encoding="utf-8",
    )


def load_name_set(path: str | None) -> set[str]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"Image list not found: {file_path}")
    return {
        Path(line.strip()).name
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def write_manifest(
    dataset: Path,
    *,
    image_dir: Path,
    label_dir: Path,
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    included: set[str] | None,
    excluded: set[str],
    strategy: str,
    seed: int,
    val_ratio: float,
) -> None:
    records = []
    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in pairs:
            records.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "label": label_path.name,
                    "image_sha256": file_sha256(image_path),
                    "label_sha256": file_sha256(label_path),
                }
            )
    manifest = {
        "source": {
            "images": str(image_dir),
            "labels": str(label_dir),
        },
        "strategy": strategy,
        "seed": seed,
        "val_ratio": val_ratio,
        "include_count": len(included) if included is not None else None,
        "excluded_names": sorted(excluded),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "records": records,
    }
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Split labeled raw images into YOLO train/val folders.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw", help="Raw image directory.")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels", help="Raw YOLO label directory.")
    parser.add_argument("--dataset", default="datasets/parking_vehicles", help="Dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--strategy",
        choices=("temporal", "random"),
        default="temporal",
        help="Temporal keeps the newest images for validation and avoids near-duplicate leakage.",
    )
    parser.add_argument("--clear", action="store_true", help="Clear existing train/val images and labels before copying.")
    parser.add_argument(
        "--include-list",
        default=None,
        help="Optional text file containing reviewed image names. Unlisted pseudo labels are excluded.",
    )
    parser.add_argument(
        "--exclude-list",
        default="fixed_validation_images_v3.txt",
        help="Optional text file containing image names that must never enter this split.",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    dataset = Path(args.dataset)
    included = load_name_set(args.include_list) if args.include_list else None
    excluded = load_name_set(args.exclude_list) if args.exclude_list and Path(args.exclude_list).exists() else set()

    pairs = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        if included is not None and image_path.name not in included:
            continue
        if image_path.name in excluded:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise SystemExit("No eligible image/label pairs found.")
    if len(pairs) < 2:
        raise SystemExit("At least two reviewed image/label pairs are required for a train/val split.")

    if args.clear:
        for split_dir in (
            dataset / "images" / "train",
            dataset / "images" / "val",
            dataset / "labels" / "train",
            dataset / "labels" / "val",
        ):
            if split_dir.exists():
                shutil.rmtree(split_dir)

    val_count = max(1, int(len(pairs) * args.val_ratio))
    if args.strategy == "random":
        random.seed(args.seed)
        random.shuffle(pairs)
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]
    else:
        train_pairs = pairs[:-val_count]
        val_pairs = pairs[-val_count:]

    update_progress(
        phase="dataset_split",
        status="running",
        detail=f"Preparing {args.strategy} train/validation split",
        completed=0,
        total=len(pairs),
    )

    copied = 0
    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in split_pairs:
            copy_pair(
                image_path,
                label_path,
                dataset / "images" / split / image_path.name,
                dataset / "labels" / split / label_path.name,
            )
            copied += 1
            if copied % 25 == 0 or copied == len(pairs):
                update_progress(
                    phase="dataset_split",
                    status="running",
                    detail=f"Copying {args.strategy} split",
                    completed=copied,
                    total=len(pairs),
                )

    dataset.mkdir(parents=True, exist_ok=True)
    write_data_yaml(dataset)
    write_manifest(
        dataset,
        image_dir=image_dir,
        label_dir=label_dir,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        included=included,
        excluded=excluded,
        strategy=args.strategy,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    print(f"Prepared {len(train_pairs)} train images and {len(val_pairs)} val images.")
    update_progress(
        phase="dataset_split",
        status="completed",
        detail=f"Prepared {len(train_pairs)} train and {len(val_pairs)} validation images ({args.strategy})",
        completed=len(pairs),
        total=len(pairs),
    )


if __name__ == "__main__":
    main()
