import argparse
import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

from managed_parking import ManagedParkingManagement
from parking_config import (
    DEFAULTS,
    env_bool,
    env_float,
    env_int,
    env_optional,
    parse_int_list,
    redacted_source,
    validate_runtime_args,
)
from parking_geometry import (
    CalibrationError,
    file_sha256,
    load_calibration,
    load_parking_regions,
    load_roi,
    reference_shape_from_calibration,
    validate_calibration,
    write_json,
)


class JsonlEventWriter:
    def __init__(self, path: str | Path, max_bytes: int, backups: int) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rotate_if_needed(self) -> None:
        if self.max_bytes <= 0 or not self.path.exists() or self.path.stat().st_size < self.max_bytes:
            return
        if self.backups <= 0:
            self.path.unlink()
            return
        for index in range(self.backups, 0, -1):
            candidate = self.path.with_suffix(self.path.suffix + f".{index}")
            previous = self.path if index == 1 else self.path.with_suffix(self.path.suffix + f".{index - 1}")
            if candidate.exists():
                candidate.unlink()
            if previous.exists():
                previous.replace(candidate)

    def write(self, event: dict[str, Any]) -> None:
        self.rotate_if_needed()
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def setup_logger(path: str | Path, max_bytes: int, backups: int) -> logging.Logger:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("parking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max(1, max_bytes),
        backupCount=max(0, backups),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"ts": time.time(), "event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))


def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Check RTSP URL, camera channel, and firewall rules.")
    return cap


def result_image(result, fallback_frame):
    plotted = getattr(result, "plot_im", None)
    if plotted is not None:
        return plotted
    im0 = getattr(result, "im0", None)
    if im0 is not None:
        return im0
    return fallback_frame


def fit_preview(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    width_scale = max_width / width if max_width > 0 else 1.0
    height_scale = max_height / height if max_height > 0 else 1.0
    scale = min(width_scale, height_scale, 1.0)
    if scale >= 1.0:
        return frame
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def extract_counts(result):
    occupied = getattr(result, "filled_slots", None)
    available = getattr(result, "available_slots", None)
    total = getattr(result, "total_slots", None)
    if total is None and occupied is not None and available is not None:
        total = occupied + available
    return {
        "total": total,
        "occupied": occupied,
        "available": available,
    }


def session_output_path(output: str | None, session_index: int) -> Path | None:
    if not output:
        return None
    path = Path(output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}_s{session_index:03d}{path.suffix}")


def validate_calibration_files(args, calibration: dict[str, Any]) -> None:
    checks = (
        ("regions_sha256", args.regions),
        ("roi_sha256", args.roi),
    )
    for hash_key, path_value in checks:
        expected = calibration.get(hash_key)
        if not expected or not path_value:
            continue
        actual = file_sha256(path_value)
        if actual.lower() != str(expected).lower():
            raise CalibrationError(f"{path_value} does not match {hash_key} in {args.calibration}.")


def build_parking(args, frame_shape: tuple[int, int], calibration: dict[str, Any]) -> ManagedParkingManagement:
    reference_shape = reference_shape_from_calibration(calibration)
    return ManagedParkingManagement(
        model=args.model,
        json_file=args.regions,
        roi_file=args.roi,
        frame_shape=frame_shape,
        calibration_reference_shape=reference_shape,
        duplicate_iou=args.duplicate_iou,
        duplicate_center_ratio=args.duplicate_center_ratio,
        duplicate_size_ratio=args.duplicate_size_ratio,
        slot_overlap_threshold=args.slot_overlap_threshold,
        one_vehicle_one_space=args.one_vehicle_one_space,
        show_uncertain_slots=args.show_uncertain_slots,
        empty_confirmations=args.empty_confirmations,
        occupied_confirmations=args.occupied_confirmations,
        classes=parse_int_list(args.classes),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
        show=False,
    )


def write_health(path: str | Path, data: dict[str, Any]) -> None:
    health_path = Path(path)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = health_path.with_suffix(health_path.suffix + ".tmp")
    write_json(temporary, data)
    temporary.replace(health_path)


def terminal_health_payload(
    *,
    reason: str,
    detection_count: int,
    session_index: int,
    config_summary: dict[str, Any],
    last_counts: dict[str, Any] | None = None,
    last_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ts": time.time(),
        "status": "stopped",
        "reason": reason,
        "session": session_index,
        "detections_completed": detection_count,
        **config_summary,
    }
    if last_counts is not None:
        payload["last_counts"] = last_counts
    if last_stats is not None:
        payload["last_stats"] = last_stats
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run periodic YOLO parking occupancy detection on a video stream.")
    parser.add_argument("--source", default=env_optional("RTSP_URL", DEFAULTS.source), help="RTSP URL, local camera index, or video file.")
    parser.add_argument("--regions", default=env_optional("PARKING_JSON", DEFAULTS.regions), help="Parking JSON file.")
    parser.add_argument("--roi", default=env_optional("PARKING_ROI", DEFAULTS.roi), help="Detection ROI JSON file.")
    parser.add_argument("--calibration", default=env_optional("CALIBRATION_FILE", DEFAULTS.calibration), help="Calibration metadata JSON file.")
    parser.add_argument("--model", default=env_optional("MODEL_PATH", DEFAULTS.model), help="YOLO model path.")
    parser.add_argument("--output", default=env_optional("OUTPUT_VIDEO", DEFAULTS.output), help="Optional output video path base.")
    parser.add_argument("--display", action="store_true", help="Show live preview window.")
    parser.add_argument("--save-jsonl", default=env_optional("SAVE_JSONL", DEFAULTS.save_jsonl), help="Write per-detection counts as rotated JSON lines.")
    parser.add_argument("--log-file", default=env_optional("LOG_FILE", DEFAULTS.log_file), help="Structured runtime log path.")
    parser.add_argument("--health-file", default=env_optional("HEALTH_FILE", DEFAULTS.health_file), help="Current health JSON path.")
    parser.add_argument("--conf", type=float, default=env_float("CONF", DEFAULTS.conf), help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=env_float("IOU", DEFAULTS.iou), help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=env_int("IMGSZ", DEFAULTS.imgsz), help="Inference image size.")
    parser.add_argument("--device", default=env_optional("DEVICE"), help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--max-det", type=int, default=env_int("MAX_DET", DEFAULTS.max_det), help="Maximum detections per frame.")
    parser.add_argument("--interval", type=float, default=env_float("DETECT_INTERVAL", DEFAULTS.interval), help="Seconds between detections. Use 0 for every frame.")
    parser.add_argument("--classes", default=env_optional("VEHICLE_CLASS_IDS", DEFAULTS.classes), help="Comma-separated class IDs. Use 0 for the custom vehicle model.")
    parser.add_argument("--reconnect-delay", type=float, default=env_float("RECONNECT_DELAY", DEFAULTS.reconnect_delay), help="Seconds to wait before reconnecting RTSP.")
    parser.add_argument("--preview-max-width", type=int, default=env_int("PREVIEW_MAX_WIDTH", DEFAULTS.preview_max_width), help="Maximum displayed preview width. Output video stays full resolution.")
    parser.add_argument("--preview-max-height", type=int, default=env_int("PREVIEW_MAX_HEIGHT", DEFAULTS.preview_max_height), help="Maximum displayed preview height. Output video stays full resolution.")
    parser.add_argument("--duplicate-iou", type=float, default=env_float("DUPLICATE_IOU", DEFAULTS.duplicate_iou))
    parser.add_argument("--duplicate-center-ratio", type=float, default=env_float("DUPLICATE_CENTER_RATIO", DEFAULTS.duplicate_center_ratio))
    parser.add_argument("--duplicate-size-ratio", type=float, default=env_float("DUPLICATE_SIZE_RATIO", DEFAULTS.duplicate_size_ratio))
    parser.add_argument("--slot-overlap-threshold", type=float, default=env_float("SLOT_OVERLAP_THRESHOLD", DEFAULTS.slot_overlap_threshold))
    parser.add_argument("--one-vehicle-one-space", action=argparse.BooleanOptionalAction, default=env_bool("ONE_VEHICLE_ONE_SPACE", DEFAULTS.one_vehicle_one_space))
    parser.add_argument("--show-uncertain-slots", action=argparse.BooleanOptionalAction, default=env_bool("SHOW_UNCERTAIN_SLOTS", DEFAULTS.show_uncertain_slots))
    parser.add_argument("--empty-confirmations", type=int, default=env_int("EMPTY_CONFIRMATIONS", DEFAULTS.empty_confirmations))
    parser.add_argument("--occupied-confirmations", type=int, default=env_int("OCCUPIED_CONFIRMATIONS", DEFAULTS.occupied_confirmations))
    parser.add_argument("--jsonl-max-bytes", type=int, default=env_int("JSONL_MAX_BYTES", DEFAULTS.jsonl_max_bytes))
    parser.add_argument("--jsonl-backups", type=int, default=env_int("JSONL_BACKUPS", DEFAULTS.jsonl_backups))
    parser.add_argument("--log-max-bytes", type=int, default=env_int("LOG_MAX_BYTES", DEFAULTS.log_max_bytes))
    parser.add_argument("--log-backups", type=int, default=env_int("LOG_BACKUPS", DEFAULTS.log_backups))
    parser.add_argument("--allow-calibration-scale", action=argparse.BooleanOptionalAction, default=env_bool("ALLOW_CALIBRATION_SCALE", DEFAULTS.allow_calibration_scale))
    parser.add_argument("--max-detections", type=int, default=None, help="Exit cleanly after this many detections. Useful for tests.")
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_runtime_args(args)
        calibration = load_calibration(args.calibration)
        validate_calibration_files(args, calibration)
    except (ValueError, CalibrationError) as exc:
        raise SystemExit(str(exc)) from exc

    logger = setup_logger(args.log_file, args.log_max_bytes, args.log_backups)
    event_writer = JsonlEventWriter(args.save_jsonl, args.jsonl_max_bytes, args.jsonl_backups)
    redacted = redacted_source(args.source)
    config_summary = {
        "mode": "periodic_detection_debounced_slots",
        "source": redacted,
        "regions": args.regions,
        "roi": args.roi,
        "calibration": args.calibration,
        "model": args.model,
        "classes": args.classes,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "interval": args.interval,
        "slot_overlap_threshold": args.slot_overlap_threshold,
        "one_vehicle_one_space": args.one_vehicle_one_space,
        "empty_confirmations": args.empty_confirmations,
    }
    log_event(logger, "startup", **config_summary)

    writer = None
    parking = None
    parking_shape = None
    detection_count = 0
    session_index = 0
    last_counts = None
    last_stats = None

    while True:
        cap = None
        try:
            session_index += 1
            cap = open_capture(args.source)
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                raise RuntimeError("Stream opened but the first frame could not be read.")

            height, width = first_frame.shape[:2]
            frame_shape = (height, width)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            reference_shape = reference_shape_from_calibration(calibration)
            regions = load_parking_regions(args.regions, target_shape=frame_shape, reference_shape=reference_shape)
            roi = load_roi(args.roi, target_shape=frame_shape, reference_shape=reference_shape)
            validate_calibration(
                calibration=calibration,
                frame_shape=frame_shape,
                regions=regions,
                roi=roi,
                allow_scale=args.allow_calibration_scale,
            )

            if parking is None or parking_shape != frame_shape:
                parking = build_parking(args, frame_shape, calibration)
                parking_shape = frame_shape

            output_path = session_output_path(args.output, session_index)
            if output_path and width and height:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_fps = fps if args.interval <= 0 else min(fps, 1.0 / args.interval)
                writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
            else:
                output_path = None

            log_event(
                logger,
                "stream_connected",
                session=session_index,
                frame_width=width,
                frame_height=height,
                output_video=str(output_path) if output_path else None,
            )
            write_health(
                args.health_file,
                {
                    "ts": time.time(),
                    "status": "running",
                    "session": session_index,
                    "frame": {"width": width, "height": height},
                    **config_summary,
                },
            )

            last_detection = float("-inf")
            pending_frame = first_frame
            while True:
                if pending_frame is None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("Stream read failed; reconnecting.")
                else:
                    frame = pending_frame
                    pending_frame = None

                now = time.monotonic()
                if args.interval > 0 and now - last_detection < args.interval:
                    continue
                last_detection = now

                result = parking(frame)
                annotated = result_image(result, frame)
                counts = extract_counts(result)
                stats = dict(parking.filter_stats)
                last_counts = counts
                last_stats = stats
                event = {
                    "ts": time.time(),
                    "session": session_index,
                    **counts,
                    "detected_vehicles": stats.get("kept_detections"),
                    **stats,
                }
                event_writer.write(event)

                if writer is not None:
                    writer.write(annotated)

                write_health(
                    args.health_file,
                    {
                        "ts": time.time(),
                        "status": "running",
                        "session": session_index,
                        "last_counts": counts,
                        "last_stats": stats,
                        "detections_completed": detection_count + 1,
                        **config_summary,
                    },
                )

                if args.display:
                    preview = fit_preview(annotated, args.preview_max_width, args.preview_max_height)
                    cv2.imshow("Parking Management", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        write_health(
                            args.health_file,
                            terminal_health_payload(
                                reason="display_quit",
                                detection_count=detection_count + 1,
                                session_index=session_index,
                                config_summary=config_summary,
                                last_counts=last_counts,
                                last_stats=last_stats,
                            ),
                        )
                        return

                detection_count += 1
                if args.max_detections is not None and detection_count >= args.max_detections:
                    write_health(
                        args.health_file,
                        terminal_health_payload(
                            reason="max_detections_reached",
                            detection_count=detection_count,
                            session_index=session_index,
                            config_summary=config_summary,
                            last_counts=last_counts,
                            last_stats=last_stats,
                        ),
                    )
                    return

        except KeyboardInterrupt:
            log_event(logger, "stopped_by_user", detections=detection_count)
            write_health(
                args.health_file,
                terminal_health_payload(
                    reason="keyboard_interrupt",
                    detection_count=detection_count,
                    session_index=session_index,
                    config_summary=config_summary,
                    last_counts=last_counts,
                    last_stats=last_stats,
                ),
            )
            return
        except CalibrationError as exc:
            log_event(logger, "calibration_failed", error=str(exc))
            write_health(
                args.health_file,
                {
                    "ts": time.time(),
                    "status": "failed",
                    "error": str(exc),
                    **config_summary,
                },
            )
            raise SystemExit(str(exc)) from exc
        except Exception as exc:
            log_event(logger, "stream_error", error=str(exc), reconnect_delay=args.reconnect_delay)
            write_health(
                args.health_file,
                {
                    "ts": time.time(),
                    "status": "reconnecting",
                    "error": str(exc),
                    "reconnect_delay": args.reconnect_delay,
                    **config_summary,
                },
            )
            time.sleep(args.reconnect_delay)
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            writer = None
            if args.display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
