"""RealSense D435 相机采集封装。

本模块只负责 D435 的彩色流、深度流启动，以及将深度帧对齐到彩色帧。
业务层不直接接触 ``pyrealsense2``，方便 PC 端用假对象做单元测试，
也方便 Jetson 端集中处理相机资源释放。
"""

from typing import Any, Dict, Optional, Tuple


class RealSenseCamera:
    """D435 对齐采集对象。

    参数单位：
    - ``width`` / ``height``：像素。
    - ``fps``：帧率，当前项目默认 30 FPS。
    - ``rs_module``：测试注入入口；真机运行时留空，由方法内部延迟导入。
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        rs_module: Optional[Any] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._rs = rs_module
        self._pipeline = None
        self._profile = None
        self._align = None
        self._started = False

    def describe(self) -> Dict[str, int]:
        """返回当前配置的图像分辨率和帧率。"""
        return {"width": self.width, "height": self.height, "fps": self.fps}

    def start(self) -> None:
        """启动彩色流和深度流。

        注意：Jetson 上必须先确保 D435 通过 USB3 正常识别，且
        ``pyrealsense2`` 已能导入。这里不做重试，启动失败直接向上抛出，
        由现场调试脚本显示具体错误。
        """
        rs = self._get_rs_module()
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        # 项目后续所有像素点测距都以彩色图坐标为准，因此深度必须对齐到彩色流。
        self._align = rs.align(rs.stream.color)
        self._profile = self._pipeline.start(config)
        self._started = True

    def capture_aligned(self) -> Tuple[Any, Any]:
        """采集一组对齐帧。

        返回：
        - ``color_frame``：BGR 彩色帧。
        - ``depth_frame``：已经对齐到彩色坐标系的深度帧。
        """
        if not self._started:
            raise RuntimeError("RealSense 相机尚未启动")

        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("未获取到有效的彩色帧或深度帧")
        return color_frame, depth_frame

    def stop(self) -> None:
        """停止相机管线。

        允许重复调用；这样脚本在异常退出时可以放心放在 ``finally`` 中释放资源。
        """
        if self._started:
            self._pipeline.stop()
            self._started = False

    def get_color_intrinsics(self) -> Any:
        """返回当前彩色流内参。

        返回对象来自 RealSense SDK，常用字段为 ``fx/fy/ppx/ppy``。
        """
        self._require_started()
        return (
            self._profile.get_stream(self._rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def get_depth_intrinsics(self) -> Any:
        """返回对齐前的原始深度流内参。

        该内参用于记录硬件状态；对齐后像素测距通常使用彩色流内参。
        """
        self._require_started()
        return (
            self._profile.get_stream(self._rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def get_aligned_depth_intrinsics(self) -> Any:
        """返回对齐到彩色图后的深度内参。

        深度已对齐到彩色坐标系，所以内参等同彩色流内参。
        """
        return self.get_color_intrinsics()

    def get_depth_scale(self) -> float:
        """返回深度原始整数值换算到米的比例。"""
        self._require_started()
        return self._profile.get_device().first_depth_sensor().get_depth_scale()

    def get_device_info(self) -> Dict[str, str]:
        """返回相机序列号和固件版本，用于标定记录追溯。"""
        self._require_started()
        device = self._profile.get_device()
        return {
            "serial_number": device.get_info(self._rs.camera_info.serial_number),
            "firmware_version": device.get_info(
                self._rs.camera_info.firmware_version
            ),
        }

    def _require_started(self) -> None:
        """保证相机已经启动；读取内参和设备信息前必须满足该条件。"""
        if not self._started:
            raise RuntimeError("RealSense 相机尚未启动")

    def _get_rs_module(self) -> Any:
        """延迟导入 ``pyrealsense2``，避免 PC 单元测试依赖真机 SDK。"""
        if self._rs is None:
            import pyrealsense2 as rs

            self._rs = rs
        return self._rs
