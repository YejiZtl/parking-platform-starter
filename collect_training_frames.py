import argparse
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

from training_progress import update_progress


def open_capture(source: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")
    return cap


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Collect training frames from a parking camera stream.")
    parser.add_argument("--source", default=os.getenv("RTSP_URL"), help="RTSP URL or local video path.")
    parser.add_argument("--output", default="datasets/parking_vehicles/raw", help="Output image directory.")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between saved frames.")
    parser.add_argument("--count", type=int, default=300, help="How many frames to save.")
    parser.add_argument("--fail-limit", type=int, default=3, help="Reconnect after this many consecutive read failures.")
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="Seconds to wait before reconnecting.")
    args = parser.parse_args()

    if not args.source:
        raise SystemExit("Missing --source or RTSP_URL in .env")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    update_progress(
        phase="capture",
        status="running",
        detail=f"Capturing fresh RTSP frames into {output_dir}",
        completed=0,
        total=args.count,
    )

    cap = None
    saved = 0
    failures = 0
    try:
        cap = open_capture(args.source)
        while saved < args.count:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                print(f"Read failed {failures}/{args.fail_limit}; retrying...")
                if failures >= args.fail_limit:
                    cap.release()
                    print(f"Reconnecting in {args.reconnect_delay}s...")
                    time.sleep(args.reconnect_delay)
                    cap = open_capture(args.source)
                    failures = 0
                else:
                    time.sleep(args.reconnect_delay)
                continue

            failures = 0
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"parking_{timestamp}_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            update_progress(
                phase="capture",
                status="running",
                detail=f"Saved {path.name}",
                completed=saved,
                total=args.count,
            )
            print(f"Saved {saved}/{args.count}: {path}")
            time.sleep(args.interval)
    except Exception as exc:
        update_progress(
            phase="capture",
            status="failed",
            detail=str(exc),
            completed=saved,
            total=args.count,
        )
        raise
    finally:
        if cap is not None:
            cap.release()

    update_progress(
        phase="capture",
        status="completed",
        detail=f"Captured {saved} fresh RTSP frames into {output_dir}",
        completed=saved,
        total=args.count,
    )


if __name__ == "__main__":
    main()
