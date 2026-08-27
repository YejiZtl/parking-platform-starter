import argparse
import json
import time
from pathlib import Path


DEFAULT_PROGRESS_FILE = Path("runs/parking_train/codex_progress.json")


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def progress_bar(completed, total, width: int = 30) -> str:
    if not isinstance(completed, (int, float)) or not isinstance(total, (int, float)) or total <= 0:
        return ""
    ratio = min(1.0, max(0.0, completed / total))
    filled = round(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total} ({ratio:.1%})"


def render(progress: dict) -> str:
    if not progress:
        return "No Codex training progress has been recorded yet."

    lines = [
        f"Phase: {progress.get('phase', '-')}",
        f"Status: {progress.get('status', '-')}",
        f"Detail: {progress.get('detail', '-')}",
    ]
    bar = progress_bar(progress.get("completed"), progress.get("total"))
    if bar:
        lines.append(f"Progress: {bar}")
    if progress.get("epoch") is not None:
        lines.append(f"Epoch: {progress.get('epoch')}/{progress.get('epochs', '-')}")
    if progress.get("metrics"):
        metrics = ", ".join(f"{key}={value:.5g}" if isinstance(value, (int, float)) else f"{key}={value}" for key, value in progress["metrics"].items())
        lines.append(f"Metrics: {metrics}")
    if progress.get("latest_model"):
        lines.append(f"Latest model: {progress['latest_model']}")
    if progress.get("backup"):
        lines.append(f"Pre-takeover backup: {progress['backup']}")
    lines.append(f"Updated: {progress.get('updated_at', '-')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or continuously watch Codex-managed training progress.")
    parser.add_argument("--file", default=str(DEFAULT_PROGRESS_FILE), help="Progress JSON file.")
    parser.add_argument("--watch", type=float, default=None, help="Refresh interval in seconds.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON once.")
    args = parser.parse_args()

    path = Path(args.file)
    if args.json:
        print(json.dumps(load_progress(path), ensure_ascii=False, indent=2))
        return

    last_output = None
    try:
        while True:
            output = render(load_progress(path))
            if output != last_output:
                if last_output is not None:
                    print("\n" + "=" * 60)
                print(output, flush=True)
                last_output = output
            if args.watch is None:
                return
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        print("\nStopped watching. Training is not stopped.")


if __name__ == "__main__":
    main()

