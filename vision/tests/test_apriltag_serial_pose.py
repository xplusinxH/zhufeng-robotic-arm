import ast
import unittest

from communication.tag_pose_protocol import (
    format_no_tag,
    format_tag_pose,
    is_get_tool_command,
)
from coordinate.pose_transform import (
    invert_transform,
    make_transform,
    relative_transform,
    transform_translation_mm,
)
from apriltag.pose_cache import PoseCache
from apriltag.pose_service import TagPoseService


class AprilTagSerialPoseTests(unittest.TestCase):
    def test_recognizes_get_tool_query_with_line_endings(self):
        self.assertTrue(is_get_tool_command("@GET_TOOL#"))
        self.assertTrue(is_get_tool_command(" @GET_TOOL#\r\n"))
        self.assertFalse(is_get_tool_command("@GET_TARGET#"))

    def test_formats_tool_pose_in_millimeters(self):
        frame = format_tag_pose((0.12345, -0.0567, 0.3456), age_ms=42)

        self.assertEqual(frame, "@TOOL,123.5,-56.7,345.6,42#")

    def test_formats_no_tag_response(self):
        self.assertEqual(format_no_tag(), "@NO_TAG#")

    def test_computes_tag1_translation_in_tag0_frame(self):
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        camera_to_tag0 = make_transform(identity_rotation, (0.10, 0.00, 0.50))
        camera_to_tag1 = make_transform(identity_rotation, (0.25, -0.05, 0.70))

        tag0_to_tag1 = relative_transform(camera_to_tag0, camera_to_tag1)

        self.assertEqual(transform_translation_mm(tag0_to_tag1), (150.0, -50.0, 200.0))

    def test_inverts_rigid_transform(self):
        rotation = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        transform = make_transform(rotation, (0.1, 0.2, 0.3))

        identity = _matmul(invert_transform(transform), transform)

        self.assertTrue(_matrix_close(identity, _identity4()))

    def test_pose_cache_returns_recent_pose_and_expires_old_pose(self):
        cache = PoseCache(max_age_s=0.5)
        cache.update((0.1, 0.2, 0.3), now_s=10.0)

        self.assertEqual(cache.get(now_s=10.2), ((0.1, 0.2, 0.3), 200))
        self.assertIsNone(cache.get(now_s=10.6))

    def test_service_updates_cache_when_both_tags_are_detected(self):
        service = TagPoseService(base_tag_id=0, tool_tag_id=1, max_age_s=0.5)
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        detections = {
            0: make_transform(identity_rotation, (0.10, 0.00, 0.50)),
            1: make_transform(identity_rotation, (0.25, -0.05, 0.70)),
        }

        updated = service.update_from_detections(detections, now_s=10.0)

        self.assertTrue(updated)
        cached = service.get_cached(now_s=10.1)
        self.assertEqual(cached[1], 100)
        self.assertAlmostEqual(cached[0][0], 0.15)
        self.assertAlmostEqual(cached[0][1], -0.05)
        self.assertAlmostEqual(cached[0][2], 0.2)

    def test_service_ignores_frames_missing_either_required_tag(self):
        service = TagPoseService(base_tag_id=0, tool_tag_id=1, max_age_s=0.5)

        updated = service.update_from_detections({}, now_s=10.0)

        self.assertFalse(updated)
        self.assertIsNone(service.get_cached(now_s=10.0))

    def test_service_formats_serial_response_from_cache(self):
        service = TagPoseService(base_tag_id=0, tool_tag_id=1, max_age_s=0.5)
        service.cache.update((0.15, -0.05, 0.2), now_s=10.0)

        self.assertEqual(service.format_response(now_s=10.1), "@TOOL,150.0,-50.0,200.0,100#")

    def test_service_returns_no_tag_when_cache_is_empty(self):
        service = TagPoseService(base_tag_id=0, tool_tag_id=1, max_age_s=0.5)

        self.assertEqual(service.format_response(now_s=10.1), "@NO_TAG#")

    def test_new_modules_are_python_36_compatible(self):
        for path in [
            "communication/tag_pose_protocol.py",
            "coordinate/pose_transform.py",
            "apriltag/detector.py",
            "apriltag/pose_cache.py",
            "apriltag/pose_service.py",
            "tools/serve_tag1_pose_serial.py",
        ]:
            with open(path, "r", encoding="utf-8") as source_file:
                ast.parse(source_file.read(), feature_version=(3, 6))


if __name__ == "__main__":
    unittest.main()


def _matmul(left, right):
    return [
        [
            sum(left[row][index] * right[index][col] for index in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def _identity4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_close(left, right, tolerance=1e-9):
    for row in range(4):
        for col in range(4):
            if abs(left[row][col] - right[row][col]) > tolerance:
                return False
    return True
