from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics.solutions.solutions import SolutionAnnotator

from managed_parking import ManagedParkingManagement
from parking_config import (
    DEFAULTS,
    env_bool,
    env_float,
    env_int,
    env_optional,
    redacted_source,
    validate_runtime_args,
)
from parking_geometry import (
    CalibrationError,
    load_calibration,
    load_parking_regions,
    load_roi,
    reference_shape_from_calibration,
    validate_calibration,
)
from run_parking import (
    JsonlEventWriter,
    build_parking,
    extract_counts,
    log_event,
    open_capture,
    setup_logger,
    validate_calibration_files,
    write_health,
)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"opencv": cv2.__version__}
    for package in ("ultralytics", "torch", "torchvision", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return versions


def publisher_command(
    *,
    ffmpeg_bin: str,
    width: int,
    height: int,
    fps: float,
    publish_url: str,
    encoder: str,
    bitrate: str,
) -> list[str]:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
    ]
    if encoder == "libx264":
        command.extend(["-preset", "veryfast", "-tune", "zerolatency"])
    command.extend(
        [
            "-b:v",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            bitrate,
            "-g",
            str(max(1, round(fps * 2))),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            publish_url,
        ]
    )
    return command


def start_publisher(args: argparse.Namespace, width: int, height: int, fps: float) -> subprocess.Popen:
    command = publisher_command(
        ffmpeg_bin=args.ffmpeg_bin,
        width=width,
        height=height,
        fps=fps,
        publish_url=args.publish_url,
        encoder=args.encoder,
        bitrate=args.bitrate,
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)


def stop_publisher(publisher: subprocess.Popen | None) -> None:
    if publisher is None:
        return
    if publisher.stdin is not None:
        try:
            publisher.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    try:
        publisher.wait(timeout=5)
    except subprocess.TimeoutExpired:
        publisher.terminate()
        try:
            publisher.wait(timeout=3)
        except subprocess.TimeoutExpired:
            publisher.kill()


def snapshot_state(parking: ManagedParkingManagement, result: Any) -> dict[str, Any]:
    stats = dict(parking.filter_stats)
    return {
        "slot_states": tuple(parking.slot_states),
        "uncertain_slot_ids": frozenset(stats.get("uncertain_slot_ids", [])),
        "counts": extract_counts(result),
        "filter_stats": stats,
        "detected_vehicles": stats.get("kept_detections", 0),
    }


def status_payload(
    state: dict[str, Any], frame_index: int, detection_index: int
) -> dict[str, Any]:
    uncertain_slot_ids = state["uncertain_slot_ids"]
    spaces = []
    for index, occupied in enumerate(state["slot_states"]):
        slot_id = index + 1
        if occupied is None:
            status = "unknown"
        elif occupied and slot_id in uncertain_slot_ids:
            status = "uncertain"
        elif occupied:
            status = "occupied"
        else:
            status = "available"
        spaces.append({"id": f"P{slot_id:03d}", "status": status})
    return {
        "ts": time.time(),
        "mode": "periodic_detection_debounced_slots",
        "frame": frame_index,
        "detection": detection_index,
        **state["counts"],
        "uncertain": len(uncertain_slot_ids),
        "detected_vehicles": state["detected_vehicles"],
        "spaces": spaces,
    }


def write_status_json(
    path: Path, state: dict[str, Any], frame_index: int, detection_index: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(status_payload(state, frame_index, detection_index), ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def render_live_frame(
    parking: ManagedParkingManagement,
    frame: np.ndarray,
    state: dict[str, Any] | None,
) -> np.ndarray:
    if state is None:
        return frame
    annotated = frame.copy()
    slot_states = state["slot_states"]
    uncertain_slot_ids = state["uncertain_slot_ids"]
    for index, region in enumerate(parking.json):
        polygon = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
        occupied = index < len(slot_states) and slot_states[index] is True
        line_color = parking.occ if occupied else parking.arc
        if occupied and index + 1 in uncertain_slot_ids and parking.show_uncertain_slots:
            line_color = parking.uncertain_color
        cv2.polylines(annotated, [polygon], True, line_color, 2)
    counts = state["counts"]
    info = {
        "Occupancy": counts["occupied"],
        "Available": counts["available"],
        "Uncertain": len(uncertain_slot_ids),
    }
    annotator = SolutionAnnotator(annotated, parking.line_width)
    annotator.display_analytics(annotated, info, (104, 31, 17), (255, 255, 255), 10)
    if parking.parking_roi is not None:
        cv2.polylines(annotated, [parking.parking_roi], True, (0, 0, 255), 4, cv2.LINE_AA)
    return annotator.result()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run periodic parking detection and publish annotated frames through FFmpeg."
    )
    parser.add_argument("--source", default=env_optional("RTSP_URL", DEFAULTS.source))
    parser.add_argument("--regions", default=env_optional("PARKING_JSON", DEFAULTS.regions))
    parser.add_argument("--roi", default=env_optional("PARKING_ROI", DEFAULTS.roi))
    parser.add_argument("--calibration", default=env_optional("CALIBRATION_FILE", DEFAULTS.calibration))
    parser.add_argument("--model", default=env_optional("MODEL_PATH", DEFAULTS.model))
    parser.add_argument("--publish-url", default=env_optional("OUTPUT_STREAM_URL", "rtsp://127.0.0.1:8554/parking"))
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=env_bool("STREAM_PUBLISH", True))
    parser.add_argument("--every-n-frames", type=int, default=env_int("DETECT_EVERY_N_FRAMES", 5))
    parser.add_argument("--output-fps", type=float, default=env_float("OUTPUT_STREAM_FPS", 0.0))
    parser.add_argument("--encoder", default=env_optional("STREAM_ENCODER", "libx264"))
    parser.add_argument("--bitrate", default=env_optional("STREAM_BITRATE", "4M"))
    parser.add_argument("--ffmpeg-bin", default=env_optional("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--save-jsonl", default=env_optional("SAVE_JSONL", DEFAULTS.save_jsonl))
    parser.add_argument("--status-json", default=env_optional("STATUS_JSON", "parking-web/data/status.json"))
    parser.add_argument("--log-file", default=env_optional("LOG_FILE", DEFAULTS.log_file))
    parser.add_argument("--health-file", default=env_optional("HEALTH_FILE", DEFAULTS.health_file))
    parser.add_argument("--conf", type=float, default=env_float("CONF", DEFAULTS.conf))
    parser.add_argument("--iou", type=float, default=env_float("IOU", DEFAULTS.iou))
    parser.add_argument("--imgsz", type=int, default=env_int("IMGSZ", DEFAULTS.imgsz))
    parser.add_argument("--device", default=env_optional("DEVICE"))
    parser.add_argument("--max-det", type=int, default=env_int("MAX_DET", DEFAULTS.max_det))
    parser.add_argument("--classes", default=env_optional("VEHICLE_CLASS_IDS", DEFAULTS.classes))
    parser.add_argument("--reconnect-delay", type=float, default=env_float("RECONNECT_DELAY", DEFAULTS.reconnect_delay))
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
    parser.add_argument("--max-detections", type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    args.interval = 0.0
    validate_runtime_args(args)
    if args.every_n_frames < 1:
        raise ValueError("--every-n-frames must be at least 1.")
    if args.output_fps < 0:
        raise ValueError("--output-fps must be greater than or equal to 0.")
    if args.publish and shutil.which(args.ffmpeg_bin) is None:
        raise ValueError(f"FFmpeg executable not found: {args.ffmpeg_bin}")
    calibration = load_calibration(args.calibration)
    validate_calibration_files(args, calibration)
    return calibration


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        calibration = validate_args(args)
    except (ValueError, CalibrationError) as exc:
        raise SystemExit(str(exc)) from exc

    logger = setup_logger(args.log_file, args.log_max_bytes, args.log_backups)
    event_writer = JsonlEventWriter(args.save_jsonl, args.jsonl_max_bytes, args.jsonl_backups)
    config_summary = {
        "mode": "periodic_detection_debounced_slots_ffmpeg_publish",
        "source": redacted_source(args.source),
        "publish_url": redacted_source(args.publish_url) if args.publish else None,
        "publish": args.publish,
        "encoder": args.encoder if args.publish else None,
        "regions": args.regions,
        "roi": args.roi,
        "calibration": args.calibration,
        "model": args.model,
        "classes": args.classes,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "every_n_frames": args.every_n_frames,
        "slot_overlap_threshold": args.slot_overlap_threshold,
        "one_vehicle_one_space": args.one_vehicle_one_space,
        "empty_confirmations": args.empty_confirmations,
        "versions": package_versions(),
    }
    log_event(logger, "startup", **config_summary)

    parking: ManagedParkingManagement | None = None
    parking_shape: tuple[int, int] | None = None
    latest_state: dict[str, Any] | None = None
    detection_future: Future | None = None
    detection_frame_index = 0
    detection_index = 0
    frame_index = 0
    next_detection_frame = 1
    publisher: subprocess.Popen | None = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parking-detection")
    stop_reason = "stopped_by_user"

    def consume_detection() -> bool:
        nonlocal detection_future, detection_index, latest_state
        if detection_future is None:
            return False
        future = detection_future
        detection_future = None
        try:
            result = future.result()
            latest_state = snapshot_state(parking, result)
            detection_index += 1
            event = {
                "ts": time.time(),
                "input_frame": detection_frame_index,
                "detection": detection_index,
                **latest_state["counts"],
                "detected_vehicles": latest_state["detected_vehicles"],
                **latest_state["filter_stats"],
            }
            event_writer.write(event)
            write_status_json(Path(args.status_json), latest_state, detection_frame_index, detection_index)
            write_health(
                args.health_file,
                {
                    "ts": time.time(),
                    "status": "running",
                    "last_counts": latest_state["counts"],
                    "last_stats": latest_state["filter_stats"],
                    "detections_completed": detection_index,
                    **config_summary,
                },
            )
            log_event(logger, "detection", **event)
            return True
        except Exception as exc:
            log_event(logger, "detection_failed", error=str(exc), input_frame=detection_frame_index)
            write_health(
                args.health_file,
                {
                    "ts": time.time(),
                    "status": "degraded",
                    "error": str(exc),
                    "detections_completed": detection_index,
                    **config_summary,
                },
            )
            return False

    try:
        while True:
            cap = None
            try:
                cap = open_capture(args.source)
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    raise RuntimeError("Stream opened but the first frame could not be read.")
                height, width = first_frame.shape[:2]
                frame_shape = (height, width)
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

                reported_fps = cap.get(cv2.CAP_PROP_FPS)
                input_fps = reported_fps if 1.0 <= reported_fps <= 120.0 else 25.0
                output_fps = args.output_fps or input_fps
                log_event(logger, "stream_connected", width=width, height=height, input_fps=input_fps, output_fps=output_fps)
                write_health(args.health_file, {"ts": time.time(), "status": "running", "frame": {"width": width, "height": height}, **config_summary})

                pending_frame = first_frame
                while True:
                    if pending_frame is not None:
                        frame = pending_frame
                        pending_frame = None
                    else:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            if detection_future is not None:
                                consume_detection()
                                if args.max_detections is not None and detection_index >= args.max_detections:
                                    stop_reason = "max_detections_reached"
                                    return
                            raise RuntimeError("Stream read failed.")
                    frame_index += 1

                    if detection_future is not None and detection_future.done():
                        consume_detection()
                        if args.max_detections is not None and detection_index >= args.max_detections:
                            stop_reason = "max_detections_reached"
                            return

                    if detection_future is None and frame_index >= next_detection_frame:
                        detection_frame_index = frame_index
                        detection_future = executor.submit(parking, frame.copy())
                        next_detection_frame = frame_index + args.every_n_frames

                    annotated = render_live_frame(parking, frame, latest_state)
                    if args.publish:
                        if publisher is None or publisher.poll() is not None:
                            stop_publisher(publisher)
                            publisher = start_publisher(args, width, height, output_fps)
                            log_event(logger, "publisher_started", encoder=args.encoder, url=redacted_source(args.publish_url))
                        if publisher.stdin is None:
                            raise RuntimeError("FFmpeg publisher stdin is unavailable.")
                        publisher.stdin.write(annotated.tobytes())

            except KeyboardInterrupt:
                raise
            except CalibrationError:
                raise
            except Exception as exc:
                log_event(logger, "stream_error", error=str(exc), reconnect_delay=args.reconnect_delay)
                write_health(args.health_file, {"ts": time.time(), "status": "reconnecting", "error": str(exc), **config_summary})
                time.sleep(args.reconnect_delay)
            finally:
                if cap is not None:
                    cap.release()
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        log_event(logger, "stopped_by_user", detections=detection_index)
    except CalibrationError as exc:
        stop_reason = "calibration_failed"
        log_event(logger, "calibration_failed", error=str(exc))
        raise SystemExit(str(exc)) from exc
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        stop_publisher(publisher)
        terminal_status = "failed" if stop_reason == "calibration_failed" else "stopped"
        health = {
            "ts": time.time(),
            "status": terminal_status,
            "reason": stop_reason,
            "detections_completed": detection_index,
            **config_summary,
        }
        if latest_state is not None:
            health["last_counts"] = latest_state["counts"]
            health["last_stats"] = latest_state["filter_stats"]
        write_health(args.health_file, health)


if __name__ == "__main__":
    main()
