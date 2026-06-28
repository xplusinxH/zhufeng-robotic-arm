import json
from pathlib import Path
import tempfile
import unittest

from tools.capture_yolo_dataset import (
    CaptureClassState,
    build_sample_record,
    parse_classes,
    save_capture_metadata,
)


class YoloCaptureToolTests(unittest.TestCase):
    def test_capture_class_state_switches_class(self):
        state = CaptureClassState(["remote", "box", "other"])

        self.assertEqual(state.current_class(), (0, "remote"))
        self.assertEqual(state.next_class(), (1, "box"))
        self.assertEqual(state.previous_class(), (0, "remote"))
        self.assertEqual(state.select_class(2), (2, "other"))

    def test_capture_class_state_rejects_empty_classes(self):
        with self.assertRaises(ValueError):
            CaptureClassState([])

    def test_parse_classes_accepts_comma_separated_names(self):
        self.assertEqual(parse_classes("remote, box,other"), ["remote", "box", "other"])

    def test_build_sample_record_uses_class_folder(self):
        record = build_sample_record(
            output_root=Path("/data/yolo_raw"),
            class_id=1,
            class_name="box",
            image_width=640,
            image_height=480,
            timestamp="2026-06-28_10-00-00_123",
        )

        self.assertEqual(record["class_id"], 1)
        self.assertEqual(record["class_name"], "box")
        self.assertEqual(record["image_width"], 640)
        self.assertEqual(record["image_height"], 480)
        self.assertEqual(
            Path(record["image_path"]).parts[-3:],
            ("images", "box", "2026-06-28_10-00-00_123.png"),
        )

    def test_save_capture_metadata_appends_json_line_and_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = build_sample_record(
                output_root=root,
                class_id=0,
                class_name="remote",
                image_width=640,
                image_height=480,
                timestamp="2026-06-28_10-00-00_123",
            )

            save_capture_metadata(root, record, ["remote", "box"])

            metadata = (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(metadata[0])["class_name"], "remote")
            self.assertEqual((root / "classes.txt").read_text(encoding="utf-8"), "remote\nbox\n")


if __name__ == "__main__":
    unittest.main()
