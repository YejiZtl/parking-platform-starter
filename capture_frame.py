import argparse
from pathlib import Path

import cv2
from dotenv import load_dotenv
import os


def read_first_frame(source: str, output_path: Path) -> None:
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Could not read a frame from the video source. Check RTSP URL, account permissions, and network.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Saved frame to: {output_path}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Capture one frame from a parking camera stream.")
    parser.add_argument("--source", default=os.getenv("RTSP_URL"), help="RTSP URL or local video path.")
    parser.add_argument("--output", default="first_frame.jpg", help="Output image path.")
    args = parser.parse_args()

    if not args.source:
        raise SystemExit("Missing --source or RTSP_URL in .env")

    read_first_frame(args.source, Path(args.output))


if __name__ == "__main__":
    main()
