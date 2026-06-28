import unittest

from perception.yolo_depth_geometry import (
    build_yolo_depth_candidates,
    depth_roi_to_object_geometry,
)


class YoloDepthGeometryTests(unittest.TestCase):
    def test_builds_geometry_from_yolo_roi_depth_points(self):
        depth_m = [[0.50 for _u in range(8)] for _v in range(6)]
        for v in range(2, 5):
            for u in range(3, 6):
                depth_m[v][u] = 0.40
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 3.0}

        geometry = depth_roi_to_object_geometry(
            depth_m,
            bbox_pixel=(3, 2, 5, 4),
            intrinsics=intrinsics,
            min_points=4,
        )

        self.assertIsNotNone(geometry)
        self.assertEqual(geometry["depth_point_count"], 9)
        self.assertEqual(geometry["bbox_pixel"], (3, 2, 5, 4))
        self.assertAlmostEqual(geometry["center_camera_m"][2], 0.40)
        self.assertGreater(geometry["shape_3d_m"]["major_axis_m"], 0.0)
        self.assertGreater(geometry["shape_3d_m"]["minor_axis_m"], 0.0)

    def test_rejects_roi_with_too_few_valid_depth_points(self):
        depth_m = [[0.0 for _u in range(5)] for _v in range(5)]

        geometry = depth_roi_to_object_geometry(
            depth_m,
            bbox_pixel=(1, 1, 3, 3),
            intrinsics={"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0},
            min_points=3,
        )

        self.assertIsNone(geometry)

    def test_builds_candidates_from_yolo_detections_with_base_transform(self):
        depth_m = [[0.50 for _u in range(8)] for _v in range(6)]
        for v in range(2, 5):
            for u in range(3, 6):
                depth_m[v][u] = 0.40
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 3.0}
        base_from_camera = [
            [1.0, 0.0, 0.0, 0.10],
            [0.0, 1.0, 0.0, 0.20],
            [0.0, 0.0, 1.0, 0.30],
            [0.0, 0.0, 0.0, 1.00],
        ]
        detections = [
            {
                "bbox_pixel": (3, 2, 5, 4),
                "class_name": "remote",
                "score": 0.87,
            }
        ]

        candidates = build_yolo_depth_candidates(
            detections,
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=4,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["id"], 0)
        self.assertEqual(candidates[0]["source"], "yolo_depth")
        self.assertEqual(candidates[0]["class_name"], "remote")
        self.assertAlmostEqual(candidates[0]["score"], 0.87)
        self.assertAlmostEqual(candidates[0]["center_base_m"][0], 0.10, places=3)
        self.assertAlmostEqual(candidates[0]["center_base_m"][1], 0.20, places=3)
        self.assertAlmostEqual(candidates[0]["center_base_m"][2], 0.70, places=3)


if __name__ == "__main__":
    unittest.main()
