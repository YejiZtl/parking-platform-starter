import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROGRESS_FILE = Path("runs/parking_train/codex_progress.json")


def _read_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def update_progress(progress_file: str | Path = DEFAULT_PROGRESS_FILE, **changes: Any) -> dict[str, Any]:
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = _read_progress(path)
    progress.update({key: _json_value(value) for key, value in changes.items()})
    progress["updated_at"] = datetime.now().isoformat(timespec="seconds")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return progress

