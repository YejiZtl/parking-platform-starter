import unittest

import numpy as np

from managed_parking import ManagedParkingManagement


def parking_logic() -> ManagedParkingManagement:
    parking = object.__new__(ManagedParkingManagement)
    parking.slot_overlap_threshold = 0.30
    parking.confs = []
    parking.get_enclosing_box = lambda box: box
    return parking


class ManagedParkingLogicTests(unittest.TestCase):
    def test_slot_matching_assigns_vehicle_to_best_slot(self):
        parking = parking_logic()
        parking.boxes = [[10, 10, 90, 90]]
        parking.confs = [0.9]
        slots = [
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32).reshape((-1, 1, 2)),
            np.array([[80, 0], [180, 0], [180, 100], [80, 100]], dtype=np.int32).reshape((-1, 1, 2)),
        ]
        assignments = parking._match_boxes_to_regions(slots)
        self.assertEqual(set(assignments), {0})
        self.assertEqual(assignments[0]["mode"], "center")

    def test_one_vehicle_one_space_prevents_multi_slot_claim(self):
        parking = parking_logic()
        parking.boxes = [[0, 0, 180, 100]]
        parking.confs = [0.8]
        slots = [
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32).reshape((-1, 1, 2)),
            np.array([[80, 0], [180, 0], [180, 100], [80, 100]], dtype=np.int32).reshape((-1, 1, 2)),
        ]
        legacy = parking._legacy_occupied_indices(slots)
        strict = parking._match_boxes_to_regions(slots)
        self.assertEqual(legacy, {0, 1})
        self.assertEqual(len(strict), 1)

    def test_state_debounce_delays_empty_transition(self):
        parking = parking_logic()
        parking.slot_states = [True]
        parking.empty_streaks = [0]
        parking.occupied_streaks = [0]
        parking.empty_confirmations = 2
        parking.occupied_confirmations = 1
        self.assertTrue(parking._stable_state(0, False))
        self.assertFalse(parking._stable_state(0, False))


if __name__ == "__main__":
    unittest.main()
