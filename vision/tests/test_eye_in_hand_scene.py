import ast
import json
from pathlib import Path
import tempfile
import unittest

from calibration.tool_camera_io import load_tool_camera_record, save_tool_camera_record
from communication.pose_protocol import is_pose_frame, parse_pose_frame
from communication.pose_source import load_base_tool_pose_from_file
from coordinate.frame_transform import (
    compose_transform,
    make_transform_from_pose_xyzw,
    transform_point,
)
from coordinate.pose_transform import make_transform
from perception.object_cluster import cluster_candidate_points
from perception.object_fusion import build_base_height_object_candidates
from perception.table_segment import extract_points_above_base_plane
from tools.offline_eye_in_hand_debug import (
    build_offline_eye_in_hand_result,
    load_depth_matrix,
    load_intrinsics,
)
from tools.capture_eye_in_hand_debug import capture_eye_in_hand_debug


class FakeIntrinsics:
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class FakeDepthFrame:
    def __init__(self, depth_m):
        self.depth_m = depth_m

    def get_width(self):
        return len(self.depth_m[0])

    def get_height(self):
        return len(self.depth_m)

    def get_distance(self, u, v):
        return self.depth_m[v][u]


class FakeEyeInHandCamera:
    def __init__(self, depth_m):
        self.depth_m = depth_m
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def capture_aligned(self):
        return object(), FakeDepthFrame(self.depth_m)

    def get_aligned_depth_intrinsics(self):
        return FakeIntrinsics(fx=100.0, fy=100.0, ppx=1.5, ppy=1.5)

    def get_depth_scale(self):
        return 0.001

    def stop(self):
        self.stopped = True


class EyeInHandSceneTests(unittest.TestCase):
    def test_loads_tool_camera_yaml_and_builds_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_camera.yaml"
            save_tool_camera_record(
                {
                    "translation_m": (0.0, 0.0, 0.05),
                    "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
                    "captured_at": "2026-06-20T19:30:00+08:00",
                    "source": "manual_test",
                },
                path,
            )

            record = load_tool_camera_record(path)

        self.assertEqual(record["schema"], "zhufeng_tool_camera_v1")
        self.assertEqual(record["unit"], "meter")
        self.assertEqual(record["translation_m"], (0.0, 0.0, 0.05))
        self.assertEqual(record["orientation_xyzw"], (0.0, 0.0, 0.0, 1.0))
        tool_point = transform_point(record["transform"], (0.0, 0.0, 0.10))
        self.assertAlmostEqual(tool_point[0], 0.0)
        self.assertAlmostEqual(tool_point[1], 0.0)
        self.assertAlmostEqual(tool_point[2], 0.15)

    def test_loads_manual_base_tool_pose_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base_tool_pose.txt"
            path.write_text(
                "@POSE,0.100,-0.020,0.300,0.000,0.000,0.000,1.000#\n",
                encoding="utf-8",
            )

            pose = load_base_tool_pose_from_file(path)

        self.assertEqual(pose["translation_m"], (0.1, -0.02, 0.3))
        self.assertEqual(pose["orientation_xyzw"], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(
            transform_point(pose["transform"], (0.0, 0.0, 0.05)),
            (0.1, -0.02, 0.35),
        )

    def test_parses_base_tool_pose_frame_from_serial(self):
        frame = "@POSE,0.100,-0.020,0.300,0.000,0.000,0.000,1.000#"

        self.assertTrue(is_pose_frame(frame))
        parsed = parse_pose_frame(frame)

        self.assertEqual(parsed["translation_m"], (0.1, -0.02, 0.3))
        self.assertEqual(parsed["orientation_xyzw"], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(
            transform_point(parsed["transform"], (0.0, 0.0, 0.05)),
            (0.1, -0.02, 0.35),
        )

    def test_composes_base_tool_and_tool_camera_transform(self):
        base_from_tool = make_transform_from_pose_xyzw(
            0.10, 0.00, 0.20, 0.0, 0.0, 0.0, 1.0
        )
        tool_from_camera = make_transform(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            (0.00, 0.00, 0.05),
        )

        base_from_camera = compose_transform(base_from_tool, tool_from_camera)

        self.assertEqual(
            transform_point(base_from_camera, (0.0, 0.0, 0.10)),
            (0.1, 0.0, 0.35),
        )

    def test_extracts_points_above_base_plane_using_dynamic_camera_pose(self):
        depth_m = [
            [0.35, 0.35, 0.35],
            [0.35, 0.45, 0.35],
            [0.35, 0.35, 0.35],
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}
        base_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.40, 0.0, 0.0, 0.0, 1.0
        )

        points = extract_points_above_base_plane(
            depth_m,
            intrinsics,
            base_from_camera,
            min_z_base_m=0.02,
            max_z_base_m=0.10,
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["pixel"], (1, 1))
        self.assertEqual(points[0]["camera_point_m"], (0.0, 0.0, 0.45))
        self.assertAlmostEqual(points[0]["base_point_m"][0], 0.0)
        self.assertAlmostEqual(points[0]["base_point_m"][1], 0.0)
        self.assertAlmostEqual(points[0]["base_point_m"][2], 0.05)
        self.assertAlmostEqual(points[0]["height_above_table_m"], 0.05)

    def test_clusters_candidate_points_and_reports_base_center(self):
        points = [
            {
                "pixel": (10, 10),
                "camera_point_m": (0.10, 0.10, 0.40),
                "base_point_m": (0.20, 0.00, 0.05),
            },
            {
                "pixel": (11, 10),
                "camera_point_m": (0.11, 0.10, 0.40),
                "base_point_m": (0.22, 0.00, 0.05),
            },
            {
                "pixel": (90, 90),
                "camera_point_m": (0.90, 0.90, 0.40),
                "base_point_m": (0.90, 0.90, 0.05),
            },
        ]

        clusters = cluster_candidate_points(points, pixel_radius=2, min_points=2)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["bbox_pixel"], (10, 10, 11, 10))
        self.assertEqual(clusters[0]["center_base_m"], (0.21000000000000002, 0.0, 0.05))
        self.assertEqual(clusters[0]["point_count"], 2)

    def test_builds_unknown_candidates_from_base_height_filter(self):
        depth_m = [
            [0.35, 0.35, 0.35, 0.35],
            [0.35, 0.45, 0.45, 0.35],
            [0.35, 0.45, 0.45, 0.35],
            [0.35, 0.35, 0.35, 0.35],
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5}
        base_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.40, 0.0, 0.0, 0.0, 1.0
        )

        candidates = build_base_height_object_candidates(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=3,
            pixel_radius=1,
            min_z_base_m=0.02,
            max_z_base_m=0.10,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["class_name"], "unknown")
        self.assertEqual(candidates[0]["source"], "base_height")
        self.assertEqual(candidates[0]["bbox_pixel"], (1, 1, 2, 2))
        self.assertEqual(candidates[0]["point_count"], 4)
        self.assertAlmostEqual(candidates[0]["center_base_m"][2], 0.05)

    def test_builds_offline_eye_in_hand_result_from_simulated_pose(self):
        depth_m = [
            [0.35, 0.35, 0.35, 0.35],
            [0.35, 0.45, 0.45, 0.35],
            [0.35, 0.45, 0.45, 0.35],
            [0.35, 0.35, 0.35, 0.35],
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5}
        base_tool_pose = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.45, 0.0, 0.0, 0.0, 1.0
        )
        tool_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0
        )

        result = build_offline_eye_in_hand_result(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_tool=base_tool_pose,
            tool_from_camera=tool_camera,
            min_points=3,
            pixel_radius=1,
            min_z_base_m=0.02,
            max_z_base_m=0.10,
        )

        self.assertEqual(result["frame"], "base")
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["bbox_pixel"], (1, 1, 2, 2))
        self.assertAlmostEqual(result["candidates"][0]["center_base_m"][2], 0.05)
        self.assertAlmostEqual(
            transform_point(result["base_from_camera"], (0.0, 0.0, 0.45))[2],
            0.05,
        )

    def test_offline_json_readers_accept_utf8_bom_files(self):
        with tempfile.TemporaryDirectory() as directory:
            depth_path = Path(directory) / "depth.json"
            intrinsics_path = Path(directory) / "intrinsics.json"
            depth_path.write_text("[[0.45]]", encoding="utf-8-sig")
            intrinsics_path.write_text(
                '{"fx":100.0,"fy":100.0,"cx":0.0,"cy":0.0}',
                encoding="utf-8-sig",
            )

            depth = load_depth_matrix(depth_path)
            intrinsics = load_intrinsics(intrinsics_path)

        self.assertEqual(depth, [[0.45]])
        self.assertEqual(intrinsics["fx"], 100.0)

    def test_captures_one_real_camera_frame_with_simulated_pose(self):
        depth_m = [
            [0.35, 0.35, 0.35, 0.35],
            [0.35, 0.45, 0.45, 0.35],
            [0.35, 0.45, 0.45, 0.35],
            [0.35, 0.35, 0.35, 0.35],
        ]
        camera = FakeEyeInHandCamera(depth_m)
        base_from_tool = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.45, 0.0, 0.0, 0.0, 1.0
        )
        tool_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0
        )

        with tempfile.TemporaryDirectory() as directory:
            result = capture_eye_in_hand_debug(
                camera=camera,
                base_from_tool=base_from_tool,
                tool_from_camera=tool_from_camera,
                output_root=directory,
                timestamp="2026-06-20_20-00-00",
                min_points=3,
                pixel_radius=1,
                min_z_base_m=0.02,
                max_z_base_m=0.10,
            )
            result_path = Path(result["output_path"])

            saved = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(camera.started)
        self.assertTrue(camera.stopped)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["bbox_pixel"], (1, 1, 2, 2))
        self.assertAlmostEqual(result["candidates"][0]["center_base_m"][2], 0.05)
        self.assertEqual(saved["candidate_count"], 1)

    def test_eye_in_hand_modules_are_python_36_compatible(self):
        for path in [
            "communication/pose_protocol.py",
            "communication/pose_source.py",
            "calibration/tool_camera_io.py",
            "coordinate/frame_transform.py",
            "perception/table_segment.py",
            "perception/object_cluster.py",
            "perception/object_fusion.py",
            "tools/capture_eye_in_hand_debug.py",
            "tools/offline_eye_in_hand_debug.py",
        ]:
            ast.parse(
                Path(path).read_text(encoding="utf-8"),
                filename=path,
                feature_version=(3, 6),
            )


if __name__ == "__main__":
    unittest.main()
