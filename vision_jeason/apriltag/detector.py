"""AprilTag detector adapter using pupil-apriltags."""

from typing import Dict, Tuple

from coordinate.pose_transform import make_transform


class AprilTagPoseDetector:
    """Detect AprilTags and return camera-to-tag transforms."""

    def __init__(self, family: str = "tag36h11", nthreads: int = 2) -> None:
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:
            raise RuntimeError(
                "缺少 pupil_apriltags，请先在 Jetson 上安装 AprilTag 检测依赖"
            ) from exc

        self._detector = Detector(
            families=family,
            nthreads=nthreads,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def detect_camera_to_tag(
        self,
        color_bgr,
        camera_params: Tuple[float, float, float, float],
        tag_size_m: float,
    ) -> Dict[int, list]:
        """Return ``{tag_id: T_camera_tag}`` for all detected tags."""
        import cv2

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=float(tag_size_m),
        )

        transforms = {}
        for detection in detections:
            if detection.pose_R is None or detection.pose_t is None:
                continue
            transforms[int(detection.tag_id)] = make_transform(
                detection.pose_R,
                (
                    float(detection.pose_t[0][0]),
                    float(detection.pose_t[1][0]),
                    float(detection.pose_t[2][0]),
                ),
            )
        return transforms
