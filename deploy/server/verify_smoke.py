from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED = {
    "total": 122,
    "occupied": 115,
    "available": 7,
    "kept_detections": 132,
}


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "total": data.get("last_counts", {}).get("total"),
        "occupied": data.get("last_counts", {}).get("occupied"),
        "available": data.get("last_counts", {}).get("available"),
        "kept_detections": data.get("last_stats", {}).get("kept_detections"),
    }
    report = {"status": data.get("status"), "reason": data.get("reason"), **actual}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if data.get("status") != "stopped" or actual != EXPECTED:
        print(json.dumps({"expected": EXPECTED, "actual": actual}, indent=2), file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
