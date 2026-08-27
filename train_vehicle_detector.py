import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from ultralytics import YOLO

from training_progress import update_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom parking-lot vehicle detector.")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained model to fine-tune.")
    parser.add_argument("--data", default="datasets/parking_vehicles/data.yaml", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1536, help="Training image size.")
    parser.add_argument("--batch", default="-1", help="Batch size, or -1 for auto batch.")
    parser.add_argument("--device", default=None, help="Training device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--project", default="runs/parking_train", help="Output project directory.")
    parser.add_argument("--name", default="vehicle_detector", help="Training run name.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader worker processes.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer name.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="Brightness augmentation gain.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for the final N epochs.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    args = parser.parse_args()

    batch = int(args.batch) if args.batch.lstrip("-").isdigit() else args.batch
    data_path = Path(args.data).resolve()
    project_path = Path(args.project).resolve()
    model = YOLO(args.model)

    def on_fit_epoch_end(trainer):
        metrics = {}
        for key, value in getattr(trainer, "metrics", {}).items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        update_progress(
            phase="training",
            status="running",
            detail=f"Training {args.name}",
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            completed=trainer.epoch + 1,
            total=trainer.epochs,
            metrics=metrics,
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    update_progress(
        phase="training",
        status="running",
        detail=f"Loading {args.model} and starting {args.name}",
        epoch=0,
        epochs=args.epochs,
        completed=0,
        total=args.epochs,
    )

    try:
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            project=str(project_path),
            name=args.name,
            exist_ok=args.exist_ok,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            hsv_v=args.hsv_v,
            close_mosaic=args.close_mosaic,
            mosaic=args.mosaic,
            cache=False,
            workers=args.workers,
        )
    except Exception as exc:
        update_progress(
            phase="training",
            status="failed",
            detail=str(exc),
            completed=0,
            total=args.epochs,
        )
        raise

    best_model = project_path / args.name / "weights" / "best.pt"
    update_progress(
        phase="training",
        status="completed",
        detail=f"Finished {args.name}",
        epoch=args.epochs,
        epochs=args.epochs,
        completed=args.epochs,
        total=args.epochs,
        latest_model=best_model if best_model.exists() else None,
    )


if __name__ == "__main__":
    main()
