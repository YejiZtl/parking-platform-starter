import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def run_step(command):
    print("\n>", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def read_text(path: Path):
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def sorted_images(image_dir: Path):
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def image_count(image_dir: Path) -> int:
    return len(sorted_images(image_dir))


def label_path_for(label_dir: Path, image_path: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return slug or "dataset"


def default_state_dir(project: str | Path, dataset: str | Path, image_dir: str | Path) -> Path:
    dataset_path = Path(dataset).resolve()
    image_path = Path(image_dir).resolve()
    identity = f"{dataset_path}|{image_path}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return Path(project) / "state" / f"{safe_slug(dataset_path.name)}-{digest}"


def load_name_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def save_name_set(path: Path, names: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(sorted(names)) + ("\n" if names else ""),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class LabelState:
    root: Path

    @property
    def progress_file(self) -> Path:
        return self.root / "label_progress.json"

    @property
    def current_model_file(self) -> Path:
        return self.root / "current_vehicle_model.txt"

    @property
    def current_batch_file(self) -> Path:
        return self.root / "current_batch.txt"

    @property
    def verified_images_file(self) -> Path:
        return self.root / "verified_images.txt"

    @property
    def confirmed_dir(self) -> Path:
        return self.root / "confirmed_batches"

    @property
    def backups_dir(self) -> Path:
        return self.root / "backups"

    def load_progress(self) -> dict:
        if not self.progress_file.exists():
            return {}
        try:
            return json.loads(self.progress_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save_progress(self, progress: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        progress["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.progress_file.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_progress(self, **changes) -> dict:
        progress = self.load_progress()
        progress.update(changes)
        progress["state_dir"] = str(self.root)
        self.save_progress(progress)
        return progress

    def load_verified_images(self) -> set[str]:
        return load_name_set(self.verified_images_file)

    def save_verified_images(self, images: set[str]) -> None:
        save_name_set(self.verified_images_file, images)

    def reset(self) -> None:
        for path in (
            self.progress_file,
            self.current_batch_file,
            self.current_model_file,
            self.verified_images_file,
        ):
            if path.exists():
                path.unlink()

    def confirmed_file(self, batch: int, start: int, end: int) -> Path:
        return self.confirmed_dir / f"batch_{batch:03d}_{start:05d}_{end:05d}.txt"


def resolve_state(args) -> LabelState:
    root = Path(args.state_dir) if args.state_dir else default_state_dir(args.project, args.dataset, args.images)
    return LabelState(root)


def first_unverified_start(images: list[Path], verified: set[str]) -> int:
    for index, image_path in enumerate(images, start=1):
        if image_path.name not in verified:
            return index
    return len(images) + 1


def choose_window(args, state: LabelState, image_dir: Path) -> tuple[int, int, int]:
    images = sorted_images(image_dir)
    total_images = len(images)
    if args.batch is not None:
        batch = args.batch
        start = (batch - 1) * args.batch_size + 1
    else:
        start = first_unverified_start(images, state.load_verified_images())
        batch = (start - 1) // args.batch_size + 1
    if start > total_images:
        raise SystemExit("All images in this dataset are already reviewed.")
    limit = min(args.batch_size, total_images - start + 1)
    return batch, start, limit


def current_model_from_progress(state: LabelState, reset_model: bool):
    if reset_model:
        return None
    progress = state.load_progress()
    return progress.get("current_model") or read_text(state.current_model_file)


def print_progress(state: LabelState, image_dir: Path, label_dir: Path):
    progress = state.load_progress()
    total_images = image_count(image_dir)
    labeled_count = sum(
        1 for image_path in sorted_images(image_dir) if label_path_for(label_dir, image_path).exists()
    )
    verified = state.load_verified_images()

    print(f"State dir: {state.root}")
    print(f"Images: {total_images}")
    print(f"Labeled files found: {labeled_count}")
    print(f"Reviewed images eligible for training: {len(verified)}")
    if not progress:
        print("No progress file yet.")
        print("Next unreviewed image: 1")
        return

    print(f"Status: {progress.get('status', 'unknown')}")
    print(f"Active batch: {progress.get('active_batch', '-')}")
    print(f"Active range: {progress.get('active_start', '-')}-{progress.get('active_end', '-')}")
    print(f"Last trained batch: {progress.get('last_trained_batch', '-')}")
    print(f"Current model: {progress.get('current_model', '-')}")
    print(f"Updated at: {progress.get('updated_at', '-')}")


def backup_relabel_state(state: LabelState, label_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = state.backups_dir / f"relabel_{timestamp}"
    suffix = 1
    while backup_dir.exists():
        backup_dir = state.backups_dir / f"relabel_{timestamp}_{suffix}"
        suffix += 1

    backup_dir.mkdir(parents=True)
    if label_dir.exists():
        shutil.copytree(label_dir, backup_dir / "raw_labels")
    if state.root.exists():
        shutil.copytree(state.root, backup_dir / "state", ignore=shutil.ignore_patterns("backups"))
    return backup_dir


def main():
    parser = argparse.ArgumentParser(description="Batch auto-label, correct, train, and reuse the improved model.")
    parser.add_argument("--images", default="datasets/parking_vehicles/raw", help="Raw image directory.")
    parser.add_argument("--labels", default="datasets/parking_vehicles/raw_labels", help="Raw YOLO label directory.")
    parser.add_argument("--dataset", default="datasets/parking_vehicles", help="Dataset root.")
    parser.add_argument("--state-dir", default=None, help="State directory. Defaults to a dataset-scoped directory under --project/state.")
    parser.add_argument("--batch", type=int, default=None, help="1-based batch number. Default: first unreviewed image window.")
    parser.add_argument("--batch-size", type=int, default=10, help="Images per batch.")
    parser.add_argument("--seed-model", default="yolo26m.pt", help="Model used before the first custom model exists.")
    parser.add_argument("--seed-classes", default="2,3,5,7", help="Vehicle classes for the seed COCO model.")
    parser.add_argument("--auto-conf", type=float, default=0.08, help="Auto-label confidence threshold.")
    parser.add_argument("--auto-iou", type=float, default=0.35, help="Lower values remove more duplicate boxes.")
    parser.add_argument("--auto-imgsz", type=int, default=1920, help="Auto-label image size.")
    parser.add_argument("--auto-tile-size", type=int, default=0, help="Optional overlapping tile size for high-recall auto-labeling.")
    parser.add_argument("--train-imgsz", type=int, default=1536, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=25, help="Fine-tuning epochs after each corrected batch.")
    parser.add_argument("--project", default="runs/parking_train", help="Training output directory.")
    parser.add_argument("--no-train", action="store_true", help="Only auto-label and open correction UI.")
    parser.add_argument("--reset-model", action="store_true", help="Ignore the saved custom model and use the seed model.")
    parser.add_argument("--show-progress", action="store_true", help="Show saved labeling progress and exit.")
    parser.add_argument("--reset-progress", action="store_true", help="Delete saved labeling progress and exit.")
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Back up all labels, auto-label every image, and restart correction at batch 1.",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    state = resolve_state(args)

    if args.reset_progress:
        state.reset()
        print(f"Progress reset: {state.root}")
        return

    if args.show_progress:
        print_progress(state, image_dir, label_dir)
        return

    total_images = image_count(image_dir)
    if total_images == 0:
        raise SystemExit(f"No images found in {image_dir}")

    progress = state.load_progress()
    if args.relabel_all:
        saved_model = current_model_from_progress(state, args.reset_model)
        auto_model = saved_model or args.seed_model
        auto_classes = "0" if saved_model else args.seed_classes
        backup_dir = backup_relabel_state(state, label_dir)

        print(f"Backup: {backup_dir}")
        print(f"Images to relabel: {total_images}")
        print(f"Auto-label model: {auto_model}")
        print(f"Confidence: {args.auto_conf}")
        print(f"IOU: {args.auto_iou}")

        run_step([
            sys.executable,
            "auto_label_vehicles.py",
            "--images",
            args.images,
            "--labels",
            args.labels,
            "--model",
            auto_model,
            "--classes",
            auto_classes,
            "--conf",
            str(args.auto_conf),
            "--iou",
            str(args.auto_iou),
            "--imgsz",
            str(args.auto_imgsz),
            "--start",
            "1",
            "--limit",
            str(total_images),
            "--overwrite",
        ] + (["--tile-size", str(args.auto_tile_size)] if args.auto_tile_size > 0 else []))

        state.save_verified_images(set())
        state.update_progress(
            status="relabelled",
            active_batch=None,
            last_trained_batch=progress.get("last_trained_batch"),
            batch_size=args.batch_size,
            total_images=total_images,
            current_model=saved_model,
            relabel_backup=str(backup_dir),
            relabel_conf=args.auto_conf,
            relabel_iou=args.auto_iou,
        )

        print("\nAll images were relabeled successfully.")
        print(f"Old labels and progress: {backup_dir}")
        print("Manual correction will restart at the first unreviewed image.")
        return

    batch, start, limit = choose_window(args, state, image_dir)
    end = start + limit - 1
    saved_model = current_model_from_progress(state, args.reset_model)
    auto_model = saved_model or args.seed_model
    auto_classes = "0" if saved_model else args.seed_classes
    train_model = saved_model or args.seed_model

    print(f"Batch: {batch}")
    print(f"Images: {start} to {end} of {total_images}")
    print(f"State dir: {state.root}")
    print(f"Auto-label model: {auto_model}")

    state.update_progress(
        status="auto_labeling",
        active_batch=batch,
        active_start=start,
        active_end=end,
        batch_size=args.batch_size,
        total_images=total_images,
        current_model=saved_model,
    )

    run_step([
        sys.executable,
        "auto_label_vehicles.py",
        "--images",
        args.images,
        "--labels",
        args.labels,
        "--model",
        auto_model,
        "--classes",
        auto_classes,
        "--conf",
        str(args.auto_conf),
        "--iou",
        str(args.auto_iou),
        "--imgsz",
        str(args.auto_imgsz),
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--overwrite",
    ] + (["--tile-size", str(args.auto_tile_size)] if args.auto_tile_size > 0 else []))

    state.update_progress(status="correcting", active_batch=batch, active_start=start, active_end=end)
    confirmed_file = state.confirmed_file(batch, start, end)
    if confirmed_file.exists():
        confirmed_file.unlink()
    run_step([
        sys.executable,
        "simple_yolo_labeler.py",
        "--images",
        args.images,
        "--labels",
        args.labels,
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--confirmed-output",
        str(confirmed_file),
    ])

    batch_images = sorted_images(image_dir)[start - 1 : start - 1 + limit]
    batch_names = {image_path.name for image_path in batch_images}
    confirmed_names = load_name_set(confirmed_file) & batch_names
    if not confirmed_names:
        state.update_progress(
            status="awaiting_review",
            active_batch=batch,
            active_start=start,
            active_end=end,
            confirmed_in_active_batch=0,
            verified_images=len(state.load_verified_images()),
        )
        print("No images were explicitly confirmed; nothing was added to training.")
        return

    verified_images = state.load_verified_images()
    verified_images.update(confirmed_names)
    state.save_verified_images(verified_images)

    unconfirmed = sorted(batch_names - confirmed_names)
    if unconfirmed:
        state.update_progress(
            status="partially_corrected",
            active_batch=batch,
            active_start=start,
            active_end=end,
            confirmed_in_active_batch=len(confirmed_names),
            unconfirmed_in_active_batch=unconfirmed,
            verified_images=len(verified_images),
        )
        print(f"Confirmed {len(confirmed_names)} image(s).")
        print("Unconfirmed images stay out of training and will be offered again.")
        return

    state.update_progress(
        status="corrected",
        last_corrected_batch=batch,
        active_batch=None,
        verified_images=len(verified_images),
    )

    if args.no_train:
        state.current_batch_file.write_text(str(batch), encoding="utf-8")
        print("Batch corrected without training. Run again to process the next unreviewed images.")
        return

    state.update_progress(status="preparing_dataset", active_batch=batch)

    run_step([
        sys.executable,
        "prepare_dataset_split.py",
        "--images",
        args.images,
        "--labels",
        args.labels,
        "--dataset",
        args.dataset,
        "--clear",
        "--include-list",
        str(state.verified_images_file),
    ])

    run_name = f"vehicle_batch_{batch:03d}"
    state.update_progress(status="training", active_batch=batch, train_run=run_name)

    run_step([
        sys.executable,
        "train_vehicle_detector.py",
        "--model",
        train_model,
        "--data",
        str(Path(args.dataset) / "data.yaml"),
        "--imgsz",
        str(args.train_imgsz),
        "--epochs",
        str(args.epochs),
        "--project",
        args.project,
        "--name",
        run_name,
        "--exist-ok",
    ])

    best_model = Path(args.project) / run_name / "weights" / "best.pt"
    if not best_model.exists():
        raise SystemExit(f"Training finished, but best model was not found: {best_model}")

    state.current_model_file.write_text(str(best_model), encoding="utf-8")
    state.current_batch_file.write_text(str(batch), encoding="utf-8")
    state.update_progress(
        status="trained",
        active_batch=None,
        last_trained_batch=batch,
        current_model=str(best_model),
    )

    print("\nBatch complete.")
    print(f"Next auto-label model: {best_model}")
    print("Run this command again to process the next unreviewed images.")


if __name__ == "__main__":
    main()
