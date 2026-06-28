from pathlib import Path
import tempfile
import unittest

from tools.relabel_yolo_raw_capture import (
    DEFAULT_CLASS_RENAME_MAP,
    build_rebuilt_metadata_records,
    relabel_raw_capture_dataset,
)


class YoloRawCaptureRelabelTests(unittest.TestCase):
    def test_default_class_rename_map_matches_confirmed_objects(self):
        self.assertEqual(DEFAULT_CLASS_RENAME_MAP["remote"], "beer_can")
        self.assertEqual(DEFAULT_CLASS_RENAME_MAP["power_bank"], "earbud_case")
        self.assertEqual(DEFAULT_CLASS_RENAME_MAP["circuit_board"], "remote")
        self.assertEqual(DEFAULT_CLASS_RENAME_MAP["box"], "phone")
        self.assertEqual(DEFAULT_CLASS_RENAME_MAP["other_object"], "power_bank")

    def test_rebuilds_metadata_from_actual_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images" / "beer_can"
            image_dir.mkdir(parents=True)
            (image_dir / "a.png").write_bytes(b"fake")

            records = build_rebuilt_metadata_records(root, ["beer_can"])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["class_id"], 0)
            self.assertEqual(records[0]["class_name"], "beer_can")
            self.assertTrue(records[0]["image_path"].endswith("images/beer_can/a.png") or records[0]["image_path"].endswith("images\\beer_can\\a.png"))

    def test_relabels_folders_without_name_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name in ("power_bank", "other_object"):
                class_dir = root / "images" / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                (class_dir / (class_name + ".png")).write_bytes(b"fake")

            result = relabel_raw_capture_dataset(
                root,
                rename_map={
                    "power_bank": "earbud_case",
                    "other_object": "power_bank",
                },
                final_classes=["earbud_case", "power_bank"],
            )

            self.assertEqual(result["image_count"], 2)
            self.assertTrue((root / "images" / "earbud_case" / "power_bank.png").exists())
            self.assertTrue((root / "images" / "power_bank" / "other_object.png").exists())
            self.assertFalse((root / "images" / "other_object").exists())


if __name__ == "__main__":
    unittest.main()
