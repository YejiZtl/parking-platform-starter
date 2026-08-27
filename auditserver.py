from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_DIRS = {"__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".whl")
RUNTIME_SUFFIXES = (".log", ".pid", ".jsonl", ".mp4", ".avi")
TEMP_SUFFIXES = (".tmp", ".temp", ".part", ".pyc")
SOURCE_SUFFIXES = {".py", ".sh", ".cmd", ".md", ".txt", ".json", ".yaml", ".yml"}
SAFE_ENV_KEYS = {
    "MODEL_PATH",
    "PARKING_JSON",
    "PARKING_ROI",
    "CALIBRATION_FILE",
    "OUTPUT_VIDEO",
    "SAVE_JSONL",
    "LOG_FILE",
    "HEALTH_FILE",
    "STATUS_JSON",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def file_record(path: Path, root: Path, stat: os.stat_result) -> dict[str, Any]:
    return {
        "path": relative(path, root),
        "bytes": stat.st_size,
        "mib": round(stat.st_size / 1024 / 1024, 3),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def parse_safe_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in SAFE_ENV_KEYS:
            values[key] = value.strip().strip('"').strip("'")
    return values


def process_info(root: Path) -> dict[str, Any]:
    pid_file = root / "logs" / "parking_rtsp.pid"
    result: dict[str, Any] = {"pid_file": relative(pid_file, root), "pid": None, "running": False}
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (FileNotFoundError, ValueError):
        return result
    result["pid"] = pid
    proc = Path("/proc") / str(pid)
    result["running"] = proc.exists()
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        result["command"] = re.sub(r"rtsp://\S+", "rtsp://***", command).strip()
    except OSError:
        pass
    return result


def duplicate_groups(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        try:
            hashes[sha256(path)].append(path)
        except OSError:
            continue
    groups = []
    for digest, members in hashes.items():
        if len(members) < 2:
            continue
        size = members[0].stat().st_size
        groups.append(
            {
                "sha256": digest,
                "bytes_each": size,
                "potential_reclaim_bytes": size * (len(members) - 1),
                "paths": [relative(path, root) for path in sorted(members)],
            }
        )
    return sorted(groups, key=lambda item: item["potential_reclaim_bytes"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only parking server storage audit.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir() or str(root) in {"/", "/home", "/home/parkuser"}:
        raise SystemExit(f"Refusing unexpected project root: {root}")

    records = []
    top_level_bytes: dict[str, int] = defaultdict(int)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_candidates = []
    model_size_groups: dict[int, list[Path]] = defaultdict(list)
    symlinks = []
    errors = []

    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(dirnames):
            candidate = directory_path / name
            if candidate.is_symlink():
                symlinks.append({"path": relative(candidate, root), "target": os.readlink(candidate)})
                dirnames.remove(name)
        for name in filenames:
            path = directory_path / name
            try:
                if path.is_symlink():
                    symlinks.append({"path": relative(path, root), "target": os.readlink(path)})
                    continue
                stat = path.stat()
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            record = file_record(path, root, stat)
            records.append(record)
            parts = Path(record["path"]).parts
            top_level_bytes[parts[0] if len(parts) > 1 else "[root files]"] += stat.st_size

            lower_name = name.lower()
            suffix = path.suffix.lower()
            if any(part in CACHE_DIRS for part in parts) or lower_name.endswith(TEMP_SUFFIXES):
                categories["cache_or_temp"].append(record)
            if lower_name.endswith(ARCHIVE_SUFFIXES):
                categories["archives_or_wheels"].append(record)
            if lower_name.endswith(RUNTIME_SUFFIXES):
                categories["runtime_outputs"].append(record)
            if "backups" in parts:
                categories["backups"].append(record)
            if suffix in SOURCE_SUFFIXES and stat.st_size <= 5 * 1024 * 1024:
                source_candidates.append(path)
            if suffix == ".pt" and stat.st_size <= 300 * 1024 * 1024:
                model_size_groups[stat.st_size].append(path)

    source_size_groups: dict[int, list[Path]] = defaultdict(list)
    for path in source_candidates:
        try:
            source_size_groups[path.stat().st_size].append(path)
        except OSError:
            pass
    duplicate_source_candidates = [
        path for members in source_size_groups.values() if len(members) > 1 for path in members
    ]
    duplicate_model_candidates = [
        path for members in model_size_groups.values() if len(members) > 1 for path in members
    ]

    category_report = {}
    for name, items in categories.items():
        ordered = sorted(items, key=lambda item: item["bytes"], reverse=True)
        category_report[name] = {
            "count": len(items),
            "total_bytes": sum(item["bytes"] for item in items),
            "largest": ordered[:100],
        }

    safe_env = parse_safe_env(root / ".env")
    referenced_paths = {}
    for key, value in safe_env.items():
        if not value:
            continue
        candidate = Path(value)
        resolved = candidate if candidate.is_absolute() else root / candidate
        referenced_paths[key] = {
            "configured": value,
            "exists": resolved.exists(),
            "resolved_within_project": str(resolved.resolve()).startswith(str(root)),
        }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "top_level_bytes": dict(sorted(top_level_bytes.items(), key=lambda item: item[1], reverse=True)),
        "largest_files": sorted(records, key=lambda item: item["bytes"], reverse=True)[:150],
        "categories": category_report,
        "duplicate_source_groups": duplicate_groups(duplicate_source_candidates, root),
        "duplicate_model_groups": duplicate_groups(duplicate_model_candidates, root),
        "safe_env": safe_env,
        "referenced_paths": referenced_paths,
        "runtime_process": process_info(root),
        "symlinks": symlinks,
        "scan_errors": errors[:100],
        "note": "Inventory only. No file was modified or deleted.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
