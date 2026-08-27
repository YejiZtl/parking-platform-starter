import argparse
import re
import time
from pathlib import Path

import requests

from training_progress import update_progress


CONTENT_RANGE_TOTAL = re.compile(r"bytes\s+\d+-\d+/(\d+)")


def response_total(response: requests.Response, current_size: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    match = CONTENT_RANGE_TOTAL.fullmatch(content_range)
    if match:
        return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if not content_length:
        return None
    length = int(content_length)
    return current_size + length if response.status_code == 206 else length


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a large file with HTTP range resume and progress tracking.")
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--retries", type=int, default=50)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for attempt in range(1, args.retries + 1):
        current_size = output.stat().st_size if output.exists() else 0
        headers = {"Range": f"bytes={current_size}-"} if current_size else {}
        try:
            with session.get(args.url, headers=headers, stream=True, timeout=(30, 60)) as response:
                if response.status_code == 416:
                    print(f"Already complete: {output}")
                    return
                response.raise_for_status()

                if current_size and response.status_code != 206:
                    current_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if current_size else "wb"

                total = response_total(response, current_size)
                downloaded = current_size
                last_report = 0.0
                with output.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 2.0:
                            downloaded_mb = round(downloaded / 1_000_000)
                            total_mb = round(total / 1_000_000) if total else None
                            detail = f"Downloading CUDA PyTorch: {downloaded_mb} MB"
                            if total_mb:
                                detail += f"/{total_mb} MB"
                            update_progress(
                                phase="environment_setup",
                                status="running",
                                detail=detail,
                                completed=downloaded,
                                total=total,
                            )
                            print(detail, flush=True)
                            last_report = now

                if total is None or downloaded >= total:
                    print(f"Downloaded: {output} ({downloaded} bytes)")
                    return
        except requests.RequestException as exc:
            update_progress(
                phase="environment_setup",
                status="retrying",
                detail=f"CUDA download retry {attempt}/{args.retries}: {exc}",
            )
            print(f"Download failed on attempt {attempt}/{args.retries}: {exc}")

        time.sleep(args.retry_delay)

    raise SystemExit(f"Download did not finish after {args.retries} attempts: {output}")


if __name__ == "__main__":
    main()
