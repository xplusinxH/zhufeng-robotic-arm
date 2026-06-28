from pathlib import Path
import tempfile
import unittest

from tools.run_desktop_vision import (
    DEFAULT_CONF,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_MODEL,
    build_status_text,
    choose_model_path,
)


class DesktopVisionProgramTests(unittest.TestCase):
    def test_defaults_point_to_current_manual_model(self):
        self.assertEqual(DEFAULT_MODEL.name, "yolov8n_manual_best.engine")
        self.assertEqual(DEFAULT_FALLBACK_MODEL.name, "yolov8n_manual_best.pt")
        self.assertEqual(DEFAULT_CONF, 0.50)

    def test_choose_model_prefers_engine_then_pt_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine = root / "model.engine"
            fallback = root / "model.pt"
            fallback.write_bytes(b"pt")

            self.assertEqual(choose_model_path(None, engine, fallback), fallback)

            engine.write_bytes(b"engine")
            self.assertEqual(choose_model_path(None, engine, fallback), engine)

    def test_choose_model_honors_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit = Path(temp_dir) / "custom.pt"
            explicit.write_bytes(b"model")

            self.assertEqual(choose_model_path(explicit, Path("missing.engine"), Path("missing.pt")), explicit)

    def test_status_text_reports_detection_summary(self):
        text = build_status_text(
            {
                "candidate_count": 2,
                "grasp_count": 1,
                "timing_ms": {"total": 42.5},
            }
        )

        self.assertIn("OBJ=2", text)
        self.assertIn("GRASP=1", text)
        self.assertIn("42.5ms", text)


if __name__ == "__main__":
    unittest.main()
