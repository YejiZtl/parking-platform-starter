import tempfile
import unittest
from pathlib import Path

from run_parking_rtsp import publisher_command, status_payload, write_status_json


class RunParkingRtspTests(unittest.TestCase):
    def test_nvenc_publisher_command_preserves_mediamtx_path(self):
        command = publisher_command(
            ffmpeg_bin="ffmpeg",
            width=2560,
            height=1440,
            fps=25.0,
            publish_url="rtsp://127.0.0.1:8554/parking",
            encoder="h264_nvenc",
            bitrate="4M",
        )

        self.assertIn("h264_nvenc", command)
        self.assertEqual(command[-1], "rtsp://127.0.0.1:8554/parking")
        self.assertNotIn("tracker", " ".join(command))

    def test_status_payload_uses_detection_semantics(self):
        state = {
            "slot_states": (True, False, True),
            "uncertain_slot_ids": frozenset({3}),
            "counts": {"total": 3, "occupied": 2, "available": 1},
            "filter_stats": {"kept_detections": 2},
            "detected_vehicles": 2,
        }

        payload = status_payload(state, frame_index=8, detection_index=2)

        self.assertEqual(payload["detected_vehicles"], 2)
        self.assertEqual(payload["spaces"][0]["status"], "occupied")
        self.assertEqual(payload["spaces"][1]["status"], "available")
        self.assertEqual(payload["spaces"][2]["status"], "uncertain")

    def test_status_json_is_written_atomically(self):
        state = {
            "slot_states": (False,),
            "uncertain_slot_ids": frozenset(),
            "counts": {"total": 1, "occupied": 0, "available": 1},
            "filter_stats": {"kept_detections": 0},
            "detected_vehicles": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "status.json"
            write_status_json(path, state, frame_index=1, detection_index=1)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_name(".status.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
