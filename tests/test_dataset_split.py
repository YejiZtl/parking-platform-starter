import tempfile
import unittest
from pathlib import Path

from prepare_dataset_split import copy_pair, load_name_set, write_data_yaml, write_manifest


class DatasetSplitTests(unittest.TestCase):
    def test_data_yaml_uses_relative_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            write_data_yaml(dataset)
            self.assertIn("path: .", (dataset / "data.yaml").read_text(encoding="utf-8"))

    def test_exclude_list_reads_image_names(self):
        with tempfile.TemporaryDirectory() as directory:
            list_path = Path(directory) / "exclude.txt"
            list_path.write_text("a.jpg\nnested/b.jpg\n", encoding="utf-8")
            self.assertEqual(load_name_set(str(list_path)), {"a.jpg", "b.jpg"})

    def test_manifest_contains_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "raw"
            labels = root / "labels"
            dataset = root / "dataset"
            images.mkdir()
            labels.mkdir()
            image = images / "a.jpg"
            label = labels / "a.txt"
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            copy_pair(image, label, dataset / "images/train" / image.name, dataset / "labels/train" / label.name)
            write_manifest(
                dataset,
                image_dir=images,
                label_dir=labels,
                train_pairs=[(image, label)],
                val_pairs=[],
                included=None,
                excluded={"fixed.jpg"},
                strategy="temporal",
                seed=42,
                val_ratio=0.2,
            )
            manifest = (dataset / "dataset_manifest.json").read_text(encoding="utf-8")
            self.assertIn("image_sha256", manifest)
            self.assertIn("fixed.jpg", manifest)


if __name__ == "__main__":
    unittest.main()
