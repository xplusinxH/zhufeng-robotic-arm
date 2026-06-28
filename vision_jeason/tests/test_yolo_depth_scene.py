import unittest

from tools.run_yolo_depth_scene import (
    build_yolo_depth_scene_result,
    format_scene_protocol_frames,
)


class YoloDepthSceneTests(unittest.TestCase):
    def test_builds_scene_result_with_grasp_from_yolo_detection(self):
        depth_m = [[0.50 for _u in range(8)] for _v in range(6)]
        for v in range(2, 5):
            for u in range(3, 6):
                depth_m[v][u] = 0.40
        detections = [
            {
                "bbox_pixel": (3, 2, 5, 4),
                "class_name": "remote",
                "score": 0.90,
            }
        ]
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 3.0}
        base_from_camera = [
            [1.0, 0.0, 0.0, 0.10],
            [0.0, 1.0, 0.0, 0.20],
            [0.0, 0.0, 1.0, 0.30],
            [0.0, 0.0, 0.0, 1.00],
        ]

        result = build_yolo_depth_scene_result(
            detections=detections,
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            image_size=(8, 6),
            min_depth_points=4,
            depth_stride=1,
        )

        self.assertEqual(result["source"], "yolo_depth")
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["grasp_count"], 1)
        self.assertEqual(result["candidates"][0]["class_name"], "remote")
        self.assertAlmostEqual(result["candidates"][0]["center_base_m"][2], 0.70)
        self.assertEqual(result["grasps"][0]["id"], 0)

    def test_formats_scene_result_as_existing_protocol_frames(self):
        result = {
            "candidate_count": 1,
            "grasp_count": 1,
            "candidates": [
                {
                    "id": 0,
                    "class_name": "remote",
                    "score": 0.90,
                    "center_base_m": (0.10, 0.20, 0.70),
                    "bbox_pixel": (3, 2, 5, 4),
                    "point_count": 9,
                    "source": "yolo_depth",
                }
            ],
            "grasps": [
                {
                    "id": 0,
                    "position_base_m": (0.10, 0.20, 0.70),
                    "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
                    "width_m": 0.03,
                    "quality": 0.81,
                    "visibility": 1.0,
                    "approach": "visibility_first_top",
                }
            ],
        }

        frames = format_scene_protocol_frames(result)

        self.assertEqual(frames[0], "@OBJ,0,remote,0.90,100.0,200.0,700.0,3,2,5,4,9,yolo_depth#")
        self.assertEqual(frames[1], "@GRASP,0,100.0,200.0,700.0,0.0000,0.0000,0.0000,1.0000,30.0,0.81,1.00,visibility_first_top#")
        self.assertEqual(frames[2], "@END,2#")

    def test_formats_no_object_when_scene_is_empty(self):
        frames = format_scene_protocol_frames(
            {"candidate_count": 0, "grasp_count": 0, "candidates": [], "grasps": []}
        )

        self.assertEqual(frames, ["@NOOBJ#", "@END,0#"])


if __name__ == "__main__":
    unittest.main()
