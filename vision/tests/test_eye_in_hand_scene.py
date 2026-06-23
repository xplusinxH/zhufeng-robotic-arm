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
from perception.object_fusion import (
    build_base_height_object_candidates,
    build_depth_foreground_object_candidates,
    build_table_plane_object_candidates,
)
from perception.grasp_planner import (
    build_visibility_aware_grasp,
    estimate_visibility_score,
)
from perception.table_plane import estimate_table_plane_diagnostics
from perception.table_segment import extract_points_above_base_plane
from tools.offline_eye_in_hand_debug import (
    build_offline_eye_in_hand_result,
    load_depth_matrix,
    load_intrinsics,
)
from tools.capture_eye_in_hand_debug import (
    capture_eye_in_hand_debug,
    make_empty_live_result,
    raw_depth_to_depth_m,
)
from tools.eye_in_hand_debug_view import (
    draw_eye_in_hand_debug_overlay,
    grasp_pixel_from_candidate_roi,
)


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


class FakeDebugImage:
    def __init__(self):
        self.copy_count = 0

    def copy(self):
        self.copy_count += 1
        return FakeDebugImage()


class FakeCv2ForEyeInHandView:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self):
        self.rectangles = []
        self.circles = []
        self.texts = []

    def rectangle(self, image, start, end, color, thickness):
        self.rectangles.append((start, end, color, thickness))

    def circle(self, image, center, radius, color, thickness):
        self.circles.append((center, radius, color, thickness))

    def putText(self, image, text, origin, font, scale, color, thickness, line_type):
        self.texts.append((text, origin, color))


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

    def test_clusters_candidate_points_with_sparse_stride_pixels(self):
        points = []
        for u in (40, 48, 56):
            for v in (80, 88, 96):
                points.append(
                    {
                        "pixel": (u, v),
                        "camera_point_m": (0.10, 0.10, 0.40),
                        "base_point_m": (0.20, 0.00, 0.05),
                    }
                )
        points.append(
            {
                "pixel": (300, 300),
                "camera_point_m": (0.30, 0.30, 0.40),
                "base_point_m": (0.30, 0.30, 0.05),
            }
        )

        clusters = cluster_candidate_points(points, pixel_radius=8, min_points=5)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["bbox_pixel"], (40, 80, 56, 96))
        self.assertEqual(clusters[0]["point_count"], 9)

    def test_builds_unknown_candidates_from_base_height_filter(self):
        depth_m = [
            [0.40, 0.40, 0.40, 0.40],
            [0.40, 0.45, 0.45, 0.40],
            [0.40, 0.45, 0.45, 0.40],
            [0.40, 0.40, 0.40, 0.40],
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

    def test_builds_foreground_candidates_without_selecting_whole_table(self):
        depth_m = [
            [0.40, 0.40, 0.40, 0.40, 0.40],
            [0.40, 0.32, 0.32, 0.32, 0.40],
            [0.40, 0.32, 0.31, 0.32, 0.40],
            [0.40, 0.32, 0.32, 0.32, 0.40],
            [0.40, 0.40, 0.40, 0.40, 0.40],
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0}
        base_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.20, 0.0, 0.0, 0.0, 1.0
        )

        candidates = build_depth_foreground_object_candidates(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=5,
            pixel_radius=1,
            foreground_delta_m=0.04,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["bbox_pixel"], (1, 1, 3, 3))
        self.assertEqual(candidates[0]["source"], "depth_foreground")
        self.assertIn("shape_pixel", candidates[0])
        self.assertIn("principal_axis_pixel", candidates[0]["shape_pixel"])

    def test_estimates_table_plane_z_offset_in_base_frame(self):
        depth_m = [
            [0.45, 0.45, 0.45],
            [0.45, 0.45, 0.45],
            [0.45, 0.45, 0.45],
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}
        base_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.43, 0.0, 0.0, 0.0, 1.0
        )

        diagnostics = estimate_table_plane_diagnostics(
            depth_m,
            intrinsics,
            base_from_camera,
            stride=1,
            min_points=5,
        )

        self.assertTrue(diagnostics["valid"])
        self.assertAlmostEqual(diagnostics["table_z_offset_m"], 0.02)
        self.assertAlmostEqual(diagnostics["z_compensation_m"], -0.02)
        self.assertAlmostEqual(diagnostics["table_tilt_deg"], 0.0)

    def test_table_plane_segmentation_finds_low_object_on_slanted_table(self):
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 10.0}
        base_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.40, 0.0, 0.0, 0.0, 1.0
        )
        depth_m = []
        for v in range(20):
            row = []
            for _u in range(20):
                row.append(0.40 + 0.001 * float(v))
            depth_m.append(row)
        for v in range(7, 13):
            for u in range(6, 14):
                depth_m[v][u] -= 0.02

        candidates = build_table_plane_object_candidates(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=10,
            pixel_radius=1,
            stride=1,
            min_height_above_plane_m=0.008,
            max_height_above_plane_m=0.08,
            table_plane_min_points=80,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source"], "table_plane")
        self.assertEqual(candidates[0]["bbox_pixel"], (6, 7, 13, 12))
        self.assertEqual(candidates[0]["point_count"], 48)

    def test_table_z_compensation_updates_candidates_and_grasps(self):
        depth_m = [
            [0.40, 0.40, 0.40, 0.40],
            [0.40, 0.45, 0.45, 0.40],
            [0.40, 0.45, 0.45, 0.40],
            [0.40, 0.40, 0.40, 0.40],
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5}
        base_tool_pose = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.43, 0.0, 0.0, 0.0, 1.0
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
            min_z_base_m=0.04,
            max_z_base_m=0.10,
            enable_table_z_compensation=True,
            table_plane_min_points=8,
        )

        self.assertTrue(result["table_plane"]["valid"])
        self.assertAlmostEqual(result["table_plane"]["z_compensation_m"], -0.02)
        self.assertAlmostEqual(result["candidates"][0]["center_base_raw_m"][2], 0.07)
        self.assertAlmostEqual(result["candidates"][0]["center_base_m"][2], 0.05)
        self.assertAlmostEqual(result["grasps"][0]["position_base_m"][2], 0.05)

    def test_rejects_bottom_edge_thin_foreground_candidate(self):
        depth_m = [[0.40 for _u in range(20)] for _v in range(20)]
        for v in range(5, 12):
            for u in range(4, 12):
                depth_m[v][u] = 0.32
        for v in range(18, 20):
            for u in range(2, 18):
                depth_m[v][u] = 0.32
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 10.0}
        base_from_camera = make_transform_from_pose_xyzw(
            0.0, 0.0, -0.20, 0.0, 0.0, 0.0, 1.0
        )

        candidates = build_depth_foreground_object_candidates(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=5,
            pixel_radius=1,
            foreground_delta_m=0.04,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["bbox_pixel"], (4, 5, 11, 11))

    def test_builds_visibility_safe_grasp_candidate(self):
        candidate = {
            "id": 7,
            "score": 0.80,
            "center_base_m": (0.20, -0.03, 0.05),
            "bbox_pixel": (40, 40, 80, 80),
            "size_m": (0.04, 0.03, 0.05),
        }

        grasp = build_visibility_aware_grasp(
            candidate,
            image_size=(320, 240),
            camera_keepout_roi=(120, 80, 220, 180),
            min_visibility=0.60,
            width_margin_m=0.01,
        )

        self.assertIsNotNone(grasp)
        self.assertEqual(grasp["id"], 7)
        self.assertEqual(grasp["position_base_m"], (0.20, -0.03, 0.05))
        self.assertEqual(grasp["orientation_xyzw"], (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(grasp["width_m"], 0.05)
        self.assertAlmostEqual(grasp["quality"], 0.80)
        self.assertAlmostEqual(grasp["visibility"], 1.0)
        self.assertEqual(grasp["approach"], "visibility_first_top")

    def test_grasp_width_uses_shape_minor_axis_when_available(self):
        candidate = {
            "id": 9,
            "score": 1.0,
            "center_base_m": (0.20, -0.03, 0.05),
            "bbox_pixel": (10, 10, 110, 40),
            "size_m": (0.18, 0.04, 0.03),
            "shape_3d_m": {
                "major_axis_m": 0.18,
                "minor_axis_m": 0.04,
                "height_m": 0.03,
            },
        }

        grasp = build_visibility_aware_grasp(
            candidate,
            image_size=(320, 240),
            width_margin_m=0.01,
        )

        self.assertAlmostEqual(grasp["width_m"], 0.05)
        self.assertAlmostEqual(grasp["object_major_axis_m"], 0.18)
        self.assertAlmostEqual(grasp["object_minor_axis_m"], 0.04)

    def test_rejects_grasp_candidate_when_roi_blocks_camera_view(self):
        candidate = {
            "id": 8,
            "score": 0.90,
            "center_base_m": (0.10, 0.00, 0.04),
            "bbox_pixel": (130, 90, 210, 170),
            "size_m": (0.03, 0.03, 0.04),
        }

        visibility = estimate_visibility_score(
            candidate["bbox_pixel"],
            camera_keepout_roi=(120, 80, 220, 180),
        )
        grasp = build_visibility_aware_grasp(
            candidate,
            image_size=(320, 240),
            camera_keepout_roi=(120, 80, 220, 180),
            min_visibility=0.60,
        )

        self.assertLess(visibility, 0.60)
        self.assertIsNone(grasp)

    def test_eye_in_hand_overlay_draws_object_grasp_and_keepout_roi(self):
        result = {
            "candidate_count": 1,
            "grasp_count": 1,
            "rejected_grasp_count": 0,
            "candidates": [
                {
                    "id": 3,
                    "class_name": "unknown",
                    "score": 0.82,
                    "bbox_pixel": (20, 30, 80, 90),
                }
            ],
            "grasps": [
                {
                    "id": 3,
                    "quality": 0.76,
                    "visibility": 0.91,
                    "approach": "visibility_first_top",
                }
            ],
        }
        cv2 = FakeCv2ForEyeInHandView()

        draw_eye_in_hand_debug_overlay(
            FakeDebugImage(),
            result,
            camera_keepout_roi=(100, 120, 180, 220),
            cv2_module=cv2,
        )

        self.assertIn(((100, 120), (180, 220), (0, 0, 255), 2), cv2.rectangles)
        self.assertIn(((20, 30), (80, 90), (0, 255, 0), 2), cv2.rectangles)
        self.assertIn(((50, 60), 6, (255, 0, 0), 2), cv2.circles)
        self.assertTrue(any("OBJ 3" in item[0] for item in cv2.texts))
        self.assertTrue(any("GRASP 3" in item[0] for item in cv2.texts))

    def test_grasp_pixel_uses_matching_candidate_roi_center(self):
        candidates = [
            {"id": 1, "bbox_pixel": (10, 10, 20, 20)},
            {"id": 4, "bbox_pixel": (30, 50, 70, 90)},
        ]

        pixel = grasp_pixel_from_candidate_roi({"id": 4}, candidates)

        self.assertEqual(pixel, (50, 70))

    def test_converts_raw_depth_buffer_to_meter_matrix(self):
        raw_depth = [
            [100, 200, 0],
            [300, 400, 500],
        ]

        depth_m = raw_depth_to_depth_m(raw_depth, depth_scale=0.001)

        self.assertEqual(
            depth_m,
            [
                [0.1, 0.2, 0.0],
                [0.3, 0.4, 0.5],
            ],
        )

    def test_empty_live_result_can_be_saved_before_manual_detection(self):
        result = make_empty_live_result(
            intrinsics={"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5},
            image_size=(640, 480),
        )

        self.assertEqual(result["frame"], "base")
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["grasp_count"], 0)
        self.assertEqual(result["rejected_grasp_count"], 0)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["grasps"], [])
        self.assertEqual(result["depth_size"], {"width": 640, "height": 480})

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
            segmentation_mode="base_height",
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
                camera_keepout_roi=(100, 100, 120, 120),
                min_visibility=0.60,
                segmentation_mode="base_height",
            )
            result_path = Path(result["output_path"])

            saved = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(camera.started)
        self.assertTrue(camera.stopped)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["bbox_pixel"], (1, 1, 2, 2))
        self.assertAlmostEqual(result["candidates"][0]["center_base_m"][2], 0.05)
        self.assertEqual(result["grasp_count"], 1)
        self.assertEqual(result["rejected_grasp_count"], 0)
        self.assertAlmostEqual(result["grasps"][0]["visibility"], 1.0)
        self.assertEqual(saved["candidate_count"], 1)
        self.assertEqual(saved["grasp_count"], 1)

    def test_eye_in_hand_modules_are_python_36_compatible(self):
        for path in [
            "communication/pose_protocol.py",
            "communication/pose_source.py",
            "calibration/tool_camera_io.py",
            "coordinate/frame_transform.py",
            "perception/table_segment.py",
            "perception/object_cluster.py",
            "perception/object_fusion.py",
            "perception/grasp_planner.py",
            "perception/table_plane.py",
            "tools/capture_eye_in_hand_debug.py",
            "tools/eye_in_hand_debug_view.py",
            "tools/offline_eye_in_hand_debug.py",
        ]:
            ast.parse(
                Path(path).read_text(encoding="utf-8"),
                filename=path,
                feature_version=(3, 6),
            )


if __name__ == "__main__":
    unittest.main()
