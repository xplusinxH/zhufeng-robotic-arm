from pathlib import Path
import unittest

from tools.run_yolo_depth_deploy_test import build_deploy_test_steps


class YoloDepthDeployRunnerTests(unittest.TestCase):
    def test_builds_full_steps_when_engine_is_missing(self):
        steps = build_deploy_test_steps(
            model_path=Path("models/yolov8n_first_best.pt"),
            engine_path=Path("models/yolov8n_first_best.engine"),
            engine_exists=False,
            frames=20,
            imgsz=416,
        )

        self.assertEqual([step["name"] for step in steps], ["check", "export", "benchmark", "scene"])
        self.assertIn("check_yolo_depth_deploy.py", steps[0]["command"])
        self.assertIn("export_yolo_tensorrt.py", steps[1]["command"])
        self.assertIn("--frames 20", steps[2]["command"])
        self.assertIn("--imgsz 416", steps[2]["command"])
        self.assertIn("--protocol", steps[3]["command"])

    def test_skips_export_when_engine_exists(self):
        steps = build_deploy_test_steps(
            model_path=Path("models/yolov8n_first_best.pt"),
            engine_path=Path("models/yolov8n_first_best.engine"),
            engine_exists=True,
            frames=10,
            imgsz=640,
        )

        self.assertEqual([step["name"] for step in steps], ["check", "benchmark", "scene"])


if __name__ == "__main__":
    unittest.main()
