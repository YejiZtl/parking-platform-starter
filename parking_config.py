from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_VEHICLE_CLASS_IDS = "0"


@dataclass(frozen=True)
class RuntimeDefaults:
    source: str | None = None
    regions: str = "bounding_boxes.json"
    roi: str = "parking_roi.json"
    calibration: str = "parking_calibration.json"
    model: str = "releases/parking_vehicle_black_verified_v3/weights/best.pt"
    output: str = "logs/parking_result.mp4"
    save_jsonl: str = "logs/parking_events.jsonl"
    log_file: str = "logs/parking_runtime.log"
    health_file: str = "logs/parking_health.json"
    conf: float = 0.12
    iou: float = 0.35
    imgsz: int = 1920
    max_det: int = 500
    interval: float = 5.0
    classes: str = DEFAULT_VEHICLE_CLASS_IDS
    reconnect_delay: float = 2.0
    preview_max_width: int = 1500
    preview_max_height: int = 850
    duplicate_iou: float = 0.70
    duplicate_center_ratio: float = 0.20
    duplicate_size_ratio: float = 1.35
    slot_overlap_threshold: float = 0.30
    one_vehicle_one_space: bool = True
    show_uncertain_slots: bool = True
    empty_confirmations: int = 2
    occupied_confirmations: int = 1
    jsonl_max_bytes: int = 50 * 1024 * 1024
    jsonl_backups: int = 10
    log_max_bytes: int = 10 * 1024 * 1024
    log_backups: int = 5
    allow_calibration_scale: bool = False


DEFAULTS = RuntimeDefaults()


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_optional(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


def parse_int_list(value: str | None) -> list[int] | None:
    if value in (None, ""):
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def default_dict() -> dict[str, Any]:
    return asdict(DEFAULTS)


def validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {value}.")


def validate_runtime_args(args: Any) -> None:
    if not args.source:
        raise ValueError("Missing --source or RTSP_URL in .env.")
    if not Path(args.regions).exists():
        raise ValueError(f"Parking region file not found: {args.regions}")
    if args.roi and not Path(args.roi).exists():
        raise ValueError(f"Parking ROI file not found: {args.roi}")
    calibration = getattr(args, "calibration", None)
    if calibration and not Path(calibration).exists():
        raise ValueError(f"Calibration file not found: {calibration}")
    if not Path(args.model).exists():
        raise ValueError(f"Model file not found: {args.model}")

    for name in (
        "conf",
        "iou",
        "duplicate_iou",
        "duplicate_center_ratio",
        "slot_overlap_threshold",
    ):
        validate_threshold(name, float(getattr(args, name)))
    validate_positive("imgsz", args.imgsz)
    validate_positive("max_det", args.max_det)
    validate_positive("duplicate_size_ratio", args.duplicate_size_ratio)
    validate_positive("empty_confirmations", args.empty_confirmations)
    validate_positive("occupied_confirmations", args.occupied_confirmations)
    if args.interval < 0:
        raise ValueError("--interval must be greater than or equal to 0.")
    if args.reconnect_delay < 0:
        raise ValueError("--reconnect-delay must be greater than or equal to 0.")


def redacted_source(source: str | None) -> str | None:
    if not source:
        return None
    if "@" not in source:
        return source
    scheme, rest = source.split("://", 1) if "://" in source else ("", source)
    credential, host = rest.split("@", 1)
    if ":" in credential:
        user = credential.split(":", 1)[0]
        safe = f"{user}:***@{host}"
    else:
        safe = f"***@{host}"
    return f"{scheme}://{safe}" if scheme else safe
