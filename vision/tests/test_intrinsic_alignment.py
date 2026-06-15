import json
from pathlib import Path
import tempfile
import unittest

from calibration.intrinsic_io import build_intrinsic_record, save_intrinsic_record
from coordinate.pixel_to_3d import pixel_depth_to_camera
from tools.alignment_data import create_capture_directory, save_measurement


class FakeIntrinsics:
    width = 640
    height = 480
    fx = 615.0
    fy = 616.0
    ppx = 320.0
    ppy = 240.0
    model = "brown_conrady"
    coeffs = [0.1, 0.2, 0.3, 0.4, 0.5]


class IntrinsicAlignmentTests(unittest.TestCase):
    def test_builds_and_saves_complete_intrinsic_record(self):
        record = build_intrinsic_record(
            FakeIntrinsics(),
            FakeIntrinsics(),
            0.001,
            "243122071071",
            "05.15.01.55",
            "2026-06-15T12:00:00+08:00",
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "camera_intrinsic.json"
            save_intrinsic_record(record, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["color"]["width"], 640)
        self.assertEqual(saved["color"]["fx"], 615.0)
        self.assertEqual(saved["color"]["cx"], 320.0)
        self.assertEqual(saved["color"]["distortion_coefficients"], FakeIntrinsics.coeffs)
        self.assertEqual(saved["depth_scale_m"], 0.001)
        self.assertEqual(saved["device"]["serial_number"], "243122071071")
        self.assertEqual(saved["captured_at"], "2026-06-15T12:00:00+08:00")

    def test_converts_pixel_depth_to_camera_coordinates(self):
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

        point = pixel_depth_to_camera(420, 290, 0.5, intrinsics)

        self.assertEqual(point, (0.1, 0.05, 0.5))

    def test_returns_none_for_invalid_depth(self):
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

        self.assertIsNone(pixel_depth_to_camera(320, 240, 0.0, intrinsics))

    def test_creates_capture_directory_and_saves_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            capture_dir = create_capture_directory(
                Path(directory), "2026-06-15_12-00-00"
            )
            save_measurement(
                capture_dir / "measurement.json",
                (320, 240),
                0.5,
                (0.0, 0.0, 0.5),
            )
            saved = json.loads(
                (capture_dir / "measurement.json").read_text(encoding="utf-8")
            )

        self.assertEqual(saved["status"], "有效")
        self.assertEqual(saved["pixel"], {"u": 320, "v": 240})
        self.assertEqual(saved["camera_point_m"], {"x": 0.0, "y": 0.0, "z": 0.5})

    def test_saves_invalid_measurement_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "measurement.json"
            save_measurement(output_path, (10, 20), 0.0, None)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["status"], "无有效深度")
        self.assertIsNone(saved["camera_point_m"])


if __name__ == "__main__":
    unittest.main()
