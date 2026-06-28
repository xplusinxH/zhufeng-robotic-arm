from pathlib import Path
import tempfile
import unittest

from tools.export_yolo_dataset_from_eye_in_hand import export_yolo_dataset


class YoloDatasetExportTests(unittest.TestCase):
    def test_exports_color_images_and_empty_yolo_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "eye_in_hand_debug"
            frame = source / "2026-06-27_20-00-00"
            frame.mkdir(parents=True)
            (frame / "color.png").write_bytes(b"fake-image")
            output = root / "yolo_dataset"

            result = export_yolo_dataset(
                source_root=source,
                output_root=output,
                classes=["remote", "box"],
            )

            self.assertEqual(result["image_count"], 1)
            self.assertTrue((output / "images" / "train" / "2026-06-27_20-00-00.png").exists())
            self.assertTrue((output / "labels" / "train" / "2026-06-27_20-00-00.txt").exists())
            self.assertEqual(
                (output / "classes.txt").read_text(encoding="utf-8"),
                "remote\nbox\n",
            )
            self.assertIn("0: remote", (output / "dataset.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
