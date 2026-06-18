import ast
import json
import unittest

from communication.tag_pose_protocol import (
    format_invalid_pose_sample,
    format_no_tag,
    format_pose_sample,
    format_tag_pose,
    is_get_tag_pose_command,
    is_get_tool_command,
)
from coordinate.pose_transform import (
    invert_transform,
    make_transform,
    relative_transform,
    transform_pose_xyzw,
    transform_translation_mm,
)
from apriltag.pose_cache import PoseCache
from apriltag.pose_sample import BaseReferenceCache, robust_average_transforms
from apriltag.pose_service import TagPoseService
from tools.serve_tag1_pose_serial import _capture_pose_sample_json
from tools.tag_debug_view import build_debug_overlay_items


class AprilTagSerialPoseTests(unittest.TestCase):
    def test_recognizes_get_tool_query_with_line_endings(self):
        self.assertTrue(is_get_tool_command("@GET_TOOL#"))
        self.assertTrue(is_get_tool_command(" @GET_TOOL#\r\n"))
        self.assertFalse(is_get_tool_command("@GET_TARGET#"))

    def test_recognizes_get_tag_pose_query_with_line_endings(self):
        self.assertTrue(is_get_tag_pose_command("@GET_TAG_POSE#"))
        self.assertTrue(is_get_tag_pose_command(" @GET_TAG_POSE#\r\n"))
        self.assertFalse(is_get_tag_pose_command("@GET_TOOL#"))

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

    def test_extracts_6d_pose_as_position_and_xyzw_quaternion(self):
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        transform = make_transform(identity_rotation, (0.10, -0.20, 0.30))

        position_m, orientation_xyzw = transform_pose_xyzw(transform)

        self.assertEqual(position_m, (0.10, -0.20, 0.30))
        self.assertEqual(orientation_xyzw, (0.0, 0.0, 0.0, 1.0))

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

    def test_service_defaults_to_confirmed_tag_ids(self):
        service = TagPoseService()

        self.assertEqual(service.base_tag_id, 0)
        self.assertEqual(service.tool_tag_id, 1)

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

    def test_robust_average_uses_median_translation_and_average_orientation(self):
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        transforms = [
            make_transform(identity_rotation, (0.10, 0.20, 0.30)),
            make_transform(identity_rotation, (0.11, 0.21, 0.31)),
            make_transform(identity_rotation, (9.00, 9.00, 9.00)),
        ]

        fused = robust_average_transforms(transforms)

        self.assertEqual(transform_pose_xyzw(fused), ((0.11, 0.21, 0.31), (0.0, 0.0, 0.0, 1.0)))

    def test_base_reference_cache_fuses_recent_base_observations(self):
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        cache = BaseReferenceCache(max_items=3)
        cache.add(make_transform(identity_rotation, (0.10, 0.00, 0.50)))
        cache.add(make_transform(identity_rotation, (0.11, 0.01, 0.51)))
        cache.add(make_transform(identity_rotation, (5.00, 5.00, 5.00)))
        cache.add(make_transform(identity_rotation, (0.12, 0.02, 0.52)))

        fused = cache.get_fused()

        self.assertEqual(transform_pose_xyzw(fused), ((0.12, 0.02, 0.52), (0.0, 0.0, 0.0, 1.0)))

    def test_formats_valid_6d_pose_sample_json(self):
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        transform = make_transform(identity_rotation, (0.10, -0.20, 0.30))

        payload = json.loads(
            format_pose_sample(
                sample_id="S0001",
                seq=1,
                timestamp_jetson="2026-06-18T12:00:00.000+08:00",
                transform=transform,
                tag_size_m=0.08,
                frame_count_used=15,
                base_ref_seen=True,
                tool0_seen=True,
                base_ref_source="cached",
                decision_margin_min=42.0,
                hamming_max=0,
            )
        )

        self.assertEqual(payload["protocol"], "sukinee_tag_pose_v1")
        self.assertEqual(payload["sample_id"], "S0001")
        self.assertEqual(payload["from_frame"], "tag_base_ref")
        self.assertEqual(payload["to_frame"], "tag_tool0")
        self.assertEqual(payload["tag_family"], "tag25h9")
        self.assertEqual(payload["tag_base_ref_id"], 0)
        self.assertEqual(payload["tag_tool0_id"], 1)
        self.assertEqual(payload["position_m"], [0.1, -0.2, 0.3])
        self.assertEqual(payload["orientation_xyzw"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(payload["frame_count_used"], 15)
        self.assertTrue(payload["quality"]["both_tags_seen"])
        self.assertEqual(payload["quality"]["base_ref_source"], "cached")
        self.assertEqual(payload["quality"]["decision_margin_min"], 42.0)
        self.assertEqual(payload["quality"]["hamming_max"], 0)

    def test_formats_invalid_pose_sample_json(self):
        payload = json.loads(
            format_invalid_pose_sample(
                sample_id="S0002",
                seq=2,
                timestamp_jetson="2026-06-18T12:00:01.000+08:00",
                tag_size_m=0.08,
                frame_count_used=0,
                base_ref_seen=True,
                tool0_seen=False,
                base_ref_source="none",
            )
        )

        self.assertIsNone(payload["position_m"])
        self.assertIsNone(payload["orientation_xyzw"])
        self.assertFalse(payload["quality"]["both_tags_seen"])
        self.assertEqual(payload["quality"]["base_ref_source"], "none")
        self.assertTrue(payload["quality"]["base_ref_seen"])
        self.assertFalse(payload["quality"]["tool0_seen"])

    def test_capture_sample_uses_cached_base_reference_when_base_is_not_visible(self):
        identity_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        base_cache = BaseReferenceCache(max_items=5)
        base_cache.add(make_transform(identity_rotation, (0.10, 0.00, 0.50)))
        detector = FakeDetector(
            [
                {1: make_transform(identity_rotation, (0.25, -0.05, 0.70))},
                {1: make_transform(identity_rotation, (0.26, -0.04, 0.71))},
            ]
        )

        response, status = _capture_pose_sample_json(
            camera=FakeCamera(),
            detector=detector,
            camera_params=(1.0, 1.0, 0.0, 0.0),
            tag_size_m=0.08,
            base_tag_id=0,
            tool_tag_id=1,
            sample_frames=2,
            min_valid_frames=2,
            seq=3,
            np_module=FakeNumpy(),
            base_ref_cache=base_cache,
        )
        payload = json.loads(response)

        self.assertEqual(payload["position_m"], [0.155, -0.045, 0.205])
        self.assertEqual(status, {"base_ref_source": "cached", "last_status": "ok"})
        self.assertEqual(payload["quality"]["base_ref_source"], "cached")
        self.assertTrue(payload["quality"]["both_tags_seen"])
        self.assertFalse(payload["quality"]["base_ref_seen"])
        self.assertTrue(payload["quality"]["tool0_seen"])

    def test_builds_debug_overlay_items_from_detections_and_status(self):
        detections = [
            FakeDetection(0, [(1, 2), (3, 4), (5, 6), (7, 8)]),
            FakeDetection(1, [(10, 20), (30, 40), (50, 60), (70, 80)]),
        ]

        items = build_debug_overlay_items(
            detections=detections,
            base_tag_id=0,
            tool_tag_id=1,
            base_ref_source="cached",
            last_status="ok",
        )

        self.assertEqual(items["status_lines"], ["base_ref_source: cached", "last_status: ok"])
        self.assertEqual(items["tags"][0]["label"], "ID 0 base_ref")
        self.assertEqual(items["tags"][1]["label"], "ID 1 tool0")
        self.assertEqual(items["tags"][0]["corners"], [(1, 2), (3, 4), (5, 6), (7, 8)])

    def test_new_modules_are_python_36_compatible(self):
        for path in [
            "communication/tag_pose_protocol.py",
            "coordinate/pose_transform.py",
            "apriltag/detector.py",
            "apriltag/pose_cache.py",
            "apriltag/pose_sample.py",
            "apriltag/pose_service.py",
            "tools/tag_debug_view.py",
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


class FakeCamera:
    def capture_aligned(self):
        return FakeColorFrame(), None


class FakeColorFrame:
    def get_data(self):
        return []


class FakeDetector:
    def __init__(self, detections):
        self._detections = list(detections)

    def detect_camera_to_tag(self, color_bgr, camera_params, tag_size_m):
        return self._detections.pop(0)


class FakeNumpy:
    @staticmethod
    def asanyarray(value):
        return value


class FakeDetection:
    def __init__(self, tag_id, corners):
        self.tag_id = tag_id
        self.corners = corners
