import unittest

from run_parking import terminal_health_payload


class RunParkingHealthTests(unittest.TestCase):
    def test_terminal_health_marks_clean_stop_and_preserves_last_result(self):
        payload = terminal_health_payload(
            reason="max_detections_reached",
            detection_count=1,
            session_index=2,
            config_summary={"mode": "periodic_detection_debounced_slots"},
            last_counts={"total": 122, "occupied": 115, "available": 7},
            last_stats={"kept_detections": 132},
        )

        self.assertEqual(payload["status"], "stopped")
        self.assertEqual(payload["reason"], "max_detections_reached")
        self.assertEqual(payload["detections_completed"], 1)
        self.assertEqual(payload["last_counts"]["total"], 122)
        self.assertEqual(payload["last_stats"]["kept_detections"], 132)


if __name__ == "__main__":
    unittest.main()
