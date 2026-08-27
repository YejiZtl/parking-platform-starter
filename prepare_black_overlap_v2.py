import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_EVAL_IMAGES = {
    "parking_20260817_155020_00046.jpg",
    "parking_20260817_155935_00100.jpg",
    "parking_20260817_162311_00237.jpg",
    "parking_20260817_164646_00374.jpg",
    "parking_20260817_165849_00444.jpg",
    "parking_20260817_170136_00460.jpg",
}


def copy_pair(image: Path, label: Path, image_dir: Path, label_dir: Path, stem: str | None = None) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    output_stem = stem or image.stem
    shutil.copy2(image, image_dir / f"{output_stem}{image.suffix.lower()}")
    shutil.copy2(label, label_dir / f"{output_stem}.txt")


def safe_rebuild(path: Path, project: Path) -> None:
    resolved = path.resolve()
    allowed = (project / "datasets").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise RuntimeError(f"Refusing to clear unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def write_yaml(path: Path, dataset_root: Path, validation_dir: str) -> None:
    del dataset_root
    path.write_text(
        "\n".join(
            (
                "path: .",
                "train: images/train",
                f"val: {validation_dir}",
                "",
                "names:",
                "  0: vehicle",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the black/overlapping-vehicle v2 dataset.")
    parser.add_argument("--source", default="datasets/parking_vehicles")
    parser.add_argument("--output", default="datasets/parking_vehicles_black_overlap_v2")
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-backup", action="store_true")
    args = parser.parse_args()

    if args.hard_repeats < 1:
        raise SystemExit("--hard-repeats must be at least 1")

    project = Path(__file__).resolve().parent
    source = (project / args.source).resolve()
    output = (project / args.output).resolve()
    raw_images = source / "raw"
    raw_labels = source / "raw_labels"
    hard_dir = source / "black_overlap_batch"
    manifest = json.loads((hard_dir / "manifest.json").read_text(encoding="utf-8"))
    hard_names = {item["image"] for item in manifest["images"]}

    missing_eval = DEFAULT_EVAL_IMAGES - hard_names
    if missing_eval:
        raise SystemExit(f"Evaluation images are missing from the hard batch: {sorted(missing_eval)}")

    pairs = []
    for image in sorted(path for path in raw_images.iterdir() if path.suffix.lower() in IMAGE_EXTS):
        label = raw_labels / f"{image.stem}.txt"
        if not label.exists():
            raise SystemExit(f"Missing label: {label}")
        pairs.append((image, label))

    if not args.skip_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = project / "runs" / "parking_train" / "backups" / f"black_overlap_v2_{stamp}"
        (backup / "hard_labels").mkdir(parents=True)
        old_model = project / "runs" / "parking_train" / "vehicle_detector_managed_v1" / "weights" / "best.pt"
        if old_model.exists():
            shutil.copy2(old_model, backup / "best_before_black_overlap_v2.pt")
        env_file = project / ".env"
        if env_file.exists():
            shutil.copy2(env_file, backup / ".env.before_black_overlap_v2")
        shutil.copy2(hard_dir / "manifest.json", backup / "manifest.json")
        for name in sorted(hard_names):
            shutil.copy2(raw_labels / f"{Path(name).stem}.txt", backup / "hard_labels" / f"{Path(name).stem}.txt")
        print(f"Backup: {backup}")

    safe_rebuild(output, project)

    regular_pairs = [pair for pair in pairs if pair[0].name not in hard_names]
    hard_train_pairs = [pair for pair in pairs if pair[0].name in hard_names - DEFAULT_EVAL_IMAGES]
    eval_pairs = [pair for pair in pairs if pair[0].name in DEFAULT_EVAL_IMAGES]
    random.Random(args.seed).shuffle(regular_pairs)
    val_count = max(1, round(len(regular_pairs) * args.val_ratio))
    val_pairs = regular_pairs[:val_count]
    train_pairs = regular_pairs[val_count:]

    for image, label in train_pairs:
        copy_pair(image, label, output / "images/train", output / "labels/train")
    for image, label in val_pairs:
        copy_pair(image, label, output / "images/val", output / "labels/val")
    for image, label in hard_train_pairs:
        copy_pair(image, label, output / "images/train", output / "labels/train")
        for repeat in range(2, args.hard_repeats + 1):
            copy_pair(
                image,
                label,
                output / "images/train",
                output / "labels/train",
                stem=f"{image.stem}_hardrep{repeat:02d}",
            )
    for image, label in eval_pairs:
        copy_pair(image, label, output / "images/eval", output / "labels/eval")

    write_yaml(output / "data.yaml", output, "images/val")
    write_yaml(output / "hard_eval.yaml", output, "images/eval")
    summary = {
        "source_images": len(pairs),
        "train_regular": len(train_pairs),
        "train_hard_unique": len(hard_train_pairs),
        "hard_repeats": args.hard_repeats,
        "train_total": len(train_pairs) + len(hard_train_pairs) * args.hard_repeats,
        "validation": len(val_pairs),
        "hard_evaluation": len(eval_pairs),
        "hard_evaluation_images": sorted(DEFAULT_EVAL_IMAGES),
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
