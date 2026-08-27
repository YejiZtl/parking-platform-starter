import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from iterative_batch_label import LabelState, default_state_dir, first_unverified_start
from simple_yolo_labeler import Labeler


class LabelingStateTests(unittest.TestCase):
    def test_default_state_dir_is_dataset_scoped(self):
        first = default_state_dir("runs/parking_train", "datasets/a", "datasets/a/raw")
        second = default_state_dir("runs/parking_train", "datasets/b", "datasets/b/raw")
        self.assertNotEqual(first, second)
        self.assertIn("state", first.parts)

    def test_first_unverified_start_skips_only_confirmed_images(self):
        images = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        self.assertEqual(first_unverified_start(images, {"a.jpg"}), 2)
        self.assertEqual(first_unverified_start(images, {"a.jpg", "b.jpg", "c.jpg"}), 4)

    def test_label_state_round_trips_verified_names(self):
        with tempfile.TemporaryDirectory() as directory:
            state = LabelState(Path(directory) / "state")
            state.save_verified_images({"b.jpg", "a.jpg"})
            self.assertEqual(state.load_verified_images(), {"a.jpg", "b.jpg"})

    def test_labeler_writes_only_confirmed_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            cv2.imwrite(str(image_dir / "one.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
            cv2.imwrite(str(image_dir / "two.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
            confirmed = root / "confirmed.txt"

            labeler = Labeler(
                image_dir,
                label_dir,
                start=1,
                limit=2,
                roi_path=None,
                confirmed_output=confirmed,
            )
            labeler.open_current()
            labeler.boxes = [(1, 1, 10, 10)]
            labeler.confirm_current(labeler.images[0], 20, 20)
            labeler.write_confirmed_output()

            self.assertEqual(confirmed.read_text(encoding="utf-8").strip(), "one.jpg")
            self.assertTrue((label_dir / "one.txt").exists())
            self.assertFalse((label_dir / "two.txt").exists())


if __name__ == "__main__":
    unittest.main()
