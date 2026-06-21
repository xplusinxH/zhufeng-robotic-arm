"""AprilTag 检测器适配层。

本模块把 ``pupil_apriltags`` 的检测结果转换为项目统一使用的
``T_camera_tag`` 齐次变换矩阵。输入图像使用 OpenCV BGR 格式，
tag 尺寸单位为米。
"""

from typing import Dict, Tuple

from coordinate.pose_transform import make_transform


class AprilTagPoseDetector:
    """AprilTag 位姿检测器。

    默认使用 ``tag25h9``，与当前机械臂临时标定任务的打印 tag 保持一致。
    """

    def __init__(self, family: str = "tag25h9", nthreads: int = 2) -> None:
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
        """检测图像中的 tag，并返回 ``{tag_id: T_camera_tag}``。

        ``T_camera_tag`` 表示 tag 坐标系在相机坐标系下的位姿，平移单位为米。
        """
        detections = self.detect(color_bgr, camera_params, tag_size_m)

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

    def detect(
        self,
        color_bgr,
        camera_params: Tuple[float, float, float, float],
        tag_size_m: float,
    ):
        """返回 ``pupil_apriltags`` 原始检测对象。

        调试窗口需要角点和 ID，因此保留该低层接口；正式计算优先使用
        :meth:`detect_camera_to_tag`。
        """
        import cv2

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        return self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=float(tag_size_m),
        )
