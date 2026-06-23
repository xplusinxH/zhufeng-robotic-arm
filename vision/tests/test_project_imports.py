from pathlib import Path
import unittest

from camera.depth_utils import is_valid_depth
from camera.realsense_camera import RealSenseCamera
from communication.protocol import (
    format_end,
    format_error,
    format_grasp_candidate,
    format_no_object,
    format_no_target,
    format_object_candidate,
    format_target,
)
from coordinate.pixel_to_3d import pixel_to_camera


class ProjectImportTests(unittest.TestCase):
    def test_config_file_exists(self):
        self.assertTrue(Path("config.yaml").is_file())

    def test_camera_description(self):
        camera = RealSenseCamera()
        self.assertEqual(camera.describe(), {"width": 640, "height": 480, "fps": 30})

    def test_depth_range_validation(self):
        self.assertTrue(is_valid_depth(0.30))
        self.assertFalse(is_valid_depth(0.0))
        self.assertFalse(is_valid_depth(2.0))

    def test_pixel_to_camera_center(self):
        self.assertEqual(
            pixel_to_camera(320, 240, 0.5, 615, 615, 320, 240),
            (0.0, 0.0, 0.5),
        )

    def test_legacy_serial_protocol_frames(self):
        self.assertEqual(format_no_target(), "@NO_TARGET#")
        self.assertEqual(
            format_target("bottle", 126.4, -35.2, 280.7, 0.86),
            "@TARGET,bottle,126.4,-35.2,280.7,0.86#",
        )

    def test_object_candidate_protocol_frame(self):
        candidate = {
            "id": 3,
            "class_name": "unknown",
            "score": 0.823,
            "center_base_m": (0.215, -0.040, 0.052),
            "bbox_pixel": (120, 80, 210, 170),
            "point_count": 438,
            "source": "base_height",
        }

        frame = format_object_candidate(candidate)

        self.assertEqual(
            frame,
            "@OBJ,3,unknown,0.82,215.0,-40.0,52.0,120,80,210,170,438,base_height#",
        )

    def test_grasp_candidate_protocol_frame(self):
        grasp = {
            "id": 3,
            "position_base_m": (0.218, -0.042, 0.068),
            "orientation_xyzw": (0.0, 0.7071, 0.0, 0.7071),
            "width_m": 0.035,
            "quality": 0.764,
            "visibility": 0.913,
            "approach": "top",
        }

        frame = format_grasp_candidate(grasp)

        self.assertEqual(
            frame,
            "@GRASP,3,218.0,-42.0,68.0,0.0000,0.7071,0.0000,0.7071,35.0,0.76,0.91,top#",
        )

    def test_no_object_and_error_protocol_frames(self):
        self.assertEqual(format_no_object(), "@NOOBJ#")
        self.assertEqual(format_end(2), "@END,2#")
        self.assertEqual(
            format_error("BAD@FRAME", "missing,field#1"),
            "@ERR,BAD_FRAME,missing field 1#",
        )


if __name__ == "__main__":
    unittest.main()
