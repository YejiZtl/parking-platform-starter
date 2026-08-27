import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from parking_config import default_dict
from parking_geometry import file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable model release package.")
    parser.add_argument("--model", required=True, help="Source best.pt model path.")
    parser.add_argument("--name", required=True, help="Release directory name under releases/.")
    parser.add_argument("--evaluation", default=None, help="Optional evaluation summary JSON to copy.")
    parser.add_argument("--notes", default=None, help="Optional release notes text file to copy.")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing release directory.")
    args = parser.parse_args()

    source_model = Path(args.model)
    if not source_model.exists():
        raise SystemExit(f"Model not found: {source_model}")
    release_dir = Path("releases") / args.name
    weights_dir = release_dir / "weights"
    model_out = weights_dir / "best.pt"
    if release_dir.exists() and not args.force:
        raise SystemExit(f"Release already exists: {release_dir}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    if model_out.exists() and not args.force:
        raise SystemExit(f"Release model already exists: {model_out}")
    shutil.copy2(source_model, model_out)
    model_sha = file_sha256(model_out)
    (release_dir / "best.pt.sha256").write_text(f"{model_sha}  weights/best.pt\n", encoding="utf-8")

    copied_evaluation = None
    if args.evaluation:
        evaluation = Path(args.evaluation)
        if not evaluation.exists():
            raise SystemExit(f"Evaluation summary not found: {evaluation}")
        copied_evaluation = release_dir / evaluation.name
        shutil.copy2(evaluation, copied_evaluation)

    copied_notes = None
    if args.notes:
        notes = Path(args.notes)
        if not notes.exists():
            raise SystemExit(f"Release notes not found: {notes}")
        copied_notes = release_dir / notes.name
        shutil.copy2(notes, copied_notes)

    write_json(
        release_dir / "release_manifest.json",
        {
            "release": args.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_model": str(source_model),
            "model": "weights/best.pt",
            "sha256": model_sha,
            "evaluation_summary": copied_evaluation.name if copied_evaluation else None,
            "notes": copied_notes.name if copied_notes else None,
            "runtime_defaults": default_dict(),
            "policy": "Release directories are immutable deployment inputs; training runs remain in runs/.",
        },
    )
    print(f"Release written: {release_dir}")
    print(f"SHA256: {model_sha}")


if __name__ == "__main__":
    main()
