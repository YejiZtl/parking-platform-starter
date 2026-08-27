import argparse
import json
import shutil
from pathlib import Path

from prepare_black_overlap_v2 import DEFAULT_EVAL_IMAGES, IMAGE_EXTS, safe_rebuild, write_yaml


def copy_pair(image: Path, label: Path, image_dir: Path, label_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / image.name)
    shutil.copy2(label, label_dir / f"{image.stem}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a black-vehicle dataset from manually verified labels only."
    )
    parser.add_argument("--source", default="datasets/parking_vehicles")
    parser.add_argument(
        "--output", default="datasets/parking_vehicles_black_verified_v3"
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    source = (project / args.source).resolve()
    output = (project / args.output).resolve()
    raw_images = source / "raw"
    raw_labels = source / "raw_labels"
    hard_dir = source / "black_overlap_batch"
    manifest_path = hard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_names = {item["image"] for item in manifest["images"]}

    if not DEFAULT_EVAL_IMAGES <= verified_names:
        missing = sorted(DEFAULT_EVAL_IMAGES - verified_names)
        raise SystemExit(f"Hard evaluation images are not verified: {missing}")

    pairs = []
    for name in sorted(verified_names):
        image = raw_images / name
        label = raw_labels / f"{Path(name).stem}.txt"
        if image.suffix.lower() not in IMAGE_EXTS or not image.exists():
            raise SystemExit(f"Missing verified image: {image}")
        if not label.exists():
            raise SystemExit(f"Missing verified label: {label}")
        pairs.append((image, label))

    safe_rebuild(output, project)
    train_pairs = [pair for pair in pairs if pair[0].name not in DEFAULT_EVAL_IMAGES]
    val_pairs = [pair for pair in pairs if pair[0].name in DEFAULT_EVAL_IMAGES]

    for image, label in train_pairs:
        copy_pair(image, label, output / "images/train", output / "labels/train")
    for image, label in val_pairs:
        copy_pair(image, label, output / "images/val", output / "labels/val")

    write_yaml(output / "data.yaml", output, "images/val")
    summary = {
        "source_manifest": str(manifest_path.relative_to(project)),
        "policy": "Only manually corrected black/overlap batch labels are included.",
        "train_images": len(train_pairs),
        "validation_images": len(val_pairs),
        "train_instances": sum(
            len(label.read_text(encoding="utf-8").splitlines())
            for _, label in train_pairs
        ),
        "validation_instances": sum(
            len(label.read_text(encoding="utf-8").splitlines())
            for _, label in val_pairs
        ),
        "validation_files": sorted(image.name for image, _ in val_pairs),
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
