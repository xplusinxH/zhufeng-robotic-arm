from pathlib import Path
import unittest

from tools.benchmark_yolo_depth_jetson import (
    DEFAULT_BASE_TOOL_POSE,
    DEFAULT_TOOL_CAMERA,
    PROJECT_ROOT,
    _build_summary,
    _percentile,
)


class YoloJetsonToolTests(unittest.TestCase):
    def test_percentile_interpolates_small_samples(self):
        self.assertAlmostEqual(_percentile([10.0, 20.0, 30.0], 50.0), 20.0)
        self.assertAlmostEqual(_percentile([10.0, 20.0, 30.0], 95.0), 29.0)

    def test_build_summary_reports_timing_and_counts(self):
        summary = _build_summary(
            [
                {
                    "capture_ms": 2.0,
                    "yolo_ms": 10.0,
                    "geometry_ms": 1.0,
                    "total_ms": 13.0,
                    "detection_count": 1,
                    "candidate_count": 1,
                },
                {
                    "capture_ms": 4.0,
                    "yolo_ms": 20.0,
                    "geometry_ms": 3.0,
                    "total_ms": 27.0,
                    "detection_count": 3,
                    "candidate_count": 2,
                },
            ]
        )

        self.assertAlmostEqual(summary["total_ms"]["avg"], 20.0)
        self.assertAlmostEqual(summary["yolo_ms"]["p50"], 15.0)
        self.assertAlmostEqual(summary["geometry_ms"]["max"], 3.0)
        self.assertAlmostEqual(summary["avg_detection_count"], 2.0)
        self.assertAlmostEqual(summary["avg_candidate_count"], 1.5)

    def test_default_pose_paths_are_inside_project(self):
        self.assertEqual(Path(DEFAULT_TOOL_CAMERA), PROJECT_ROOT / "calibration" / "tool_camera.example.yaml")
        self.assertEqual(Path(DEFAULT_BASE_TOOL_POSE), PROJECT_ROOT / "tools" / "base_tool_pose.example.txt")


if __name__ == "__main__":
    unittest.main()
