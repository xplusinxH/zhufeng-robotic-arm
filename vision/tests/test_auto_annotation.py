import json
from pathlib import Path
import tempfile
import unittest

from auto_annotation.annotation_pipeline import (
    MaskPrediction,
    annotate_eye_in_hand_dataset,
    bbox_from_mask_pixels,
    filter_obvious_objects,
)


class FakeSegmentationBackend:
    def segment_image(self, image_path):
        _ = image_path
        return [
            MaskPrediction(
                mask_pixels=[(10, 20), (11, 20), (12, 21), (12, 22)],
                score=0.91,
                class_name="object",
            )
        ]


class AutoAnnotationTests(unittest.TestCase):
    def test_builds_bbox_from_mask_pixels(self):
        bbox = bbox_from_mask_pixels([(10, 20), (11, 20), (12, 21), (12, 22)])

        self.assertEqual(bbox, [10, 20, 12, 22])

    def test_annotates_eye_in_hand_dataset_with_backend_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_dir = root / "2026-06-27_17-00-00"
            frame_dir.mkdir()
            (frame_dir / "color.png").write_bytes(b"fake-color")

            result = annotate_eye_in_hand_dataset(
                dataset_root=root,
                output_annotations=root / "annotations.json",
                backend=FakeSegmentationBackend(),
                preview_writer=None,
                min_area_pixel=1,
            )
            annotations = json.loads((root / "annotations.json").read_text(encoding="utf-8"))

        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(result["object_count"], 1)
        self.assertEqual(
            annotations["frames"]["2026-06-27_17-00-00"]["objects"],
            [
                {
                    "bbox_pixel": [10, 20, 12, 22],
                    "class_name": "object",
                    "score": 0.91,
                }
            ],
        )

    def test_filters_sam_fragments_and_large_background_masks(self):
        objects = [
            {"bbox_pixel": [0, 0, 639, 479], "class_name": "object", "score": 0.99},
            {"bbox_pixel": [120, 120, 135, 132], "class_name": "object", "score": 0.99},
            {"bbox_pixel": [180, 90, 350, 260], "class_name": "object", "score": 0.98},
            {"bbox_pixel": [190, 100, 340, 250], "class_name": "object", "score": 0.97},
            {"bbox_pixel": [500, 8, 620, 90], "class_name": "object", "score": 0.96},
        ]

        filtered = filter_obvious_objects(objects, image_size=(640, 480))

        self.assertEqual(
            filtered,
            [{"bbox_pixel": [180, 90, 350, 260], "class_name": "object", "score": 0.98}],
        )


if __name__ == "__main__":
    unittest.main()
