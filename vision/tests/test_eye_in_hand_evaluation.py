import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_eye_in_hand_detection import (
    bbox_iou,
    build_annotation_template,
    evaluate_dataset,
    evaluate_frame,
)


class EyeInHandEvaluationTests(unittest.TestCase):
    def test_computes_bbox_iou(self):
        iou = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))

        self.assertAlmostEqual(iou, 25.0 / 175.0)

    def test_evaluates_frame_with_false_positive_and_false_negative(self):
        frame = evaluate_frame(
            "sample",
            detected_objects=[
                {"bbox_pixel": (0, 0, 10, 10)},
                {"bbox_pixel": (40, 40, 50, 50)},
            ],
            expected_objects=[
                {"bbox_pixel": (1, 1, 11, 11), "class_name": "remote"},
                {"bbox_pixel": (80, 80, 90, 90), "class_name": "cube"},
            ],
            iou_threshold=0.50,
        )

        self.assertEqual(frame["tp"], 1)
        self.assertEqual(frame["fp"], 1)
        self.assertEqual(frame["fn"], 1)
        self.assertEqual(len(frame["false_positives"]), 1)
        self.assertEqual(len(frame["false_negatives"]), 1)

    def test_evaluates_dataset_from_annotation_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_dir = root / "2026-06-27_17-00-00"
            frame_dir.mkdir()
            (frame_dir / "eye_in_hand_candidates.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {"id": 0, "bbox_pixel": [0, 0, 10, 10], "score": 1.0},
                            {"id": 1, "bbox_pixel": [40, 40, 50, 50], "score": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            annotations_path = root / "annotations.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "frames": {
                            "2026-06-27_17-00-00": {
                                "objects": [
                                    {"bbox_pixel": [1, 1, 11, 11], "class_name": "remote"},
                                    {"bbox_pixel": [80, 80, 90, 90], "class_name": "cube"},
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_dataset(root, annotations_path, iou_threshold=0.50)

        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["tp"], 1)
        self.assertEqual(report["fp"], 1)
        self.assertEqual(report["fn"], 1)
        self.assertAlmostEqual(report["precision"], 0.5)
        self.assertAlmostEqual(report["recall"], 0.5)
        self.assertAlmostEqual(report["f1"], 0.5)

    def test_builds_annotation_template_from_saved_frame_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame_name in ("2026-06-27_17-00-00", "2026-06-27_17-01-00"):
                frame_dir = root / frame_name
                frame_dir.mkdir()
                (frame_dir / "eye_in_hand_candidates.json").write_text(
                    json.dumps({"candidates": []}),
                    encoding="utf-8",
                )

            template = build_annotation_template(root)

        self.assertEqual(
            sorted(template["frames"].keys()),
            ["2026-06-27_17-00-00", "2026-06-27_17-01-00"],
        )
        self.assertEqual(
            template["frames"]["2026-06-27_17-00-00"],
            {"objects": []},
        )


if __name__ == "__main__":
    unittest.main()
