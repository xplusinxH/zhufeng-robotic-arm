"""RealSense D435 capture wrapper."""

from typing import Any, Dict, Optional, Tuple


class RealSenseCamera:
    """Capture aligned color and depth frames from a RealSense camera."""

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
        self._align = None
        self._started = False

    def describe(self) -> Dict[str, int]:
        """Return the configured stream shape."""
        return {"width": self.width, "height": self.height, "fps": self.fps}

    def start(self) -> None:
        """Start color and depth streams."""
        rs = self._get_rs_module()
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self._align = rs.align(rs.stream.color)
        self._pipeline.start(config)
        self._started = True

    def capture_aligned(self) -> Tuple[Any, Any]:
        """Return one color frame and its aligned depth frame."""
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
        """Stop the camera pipeline if it is running."""
        if self._started:
            self._pipeline.stop()
            self._started = False

    def _get_rs_module(self) -> Any:
        if self._rs is None:
            import pyrealsense2 as rs

            self._rs = rs
        return self._rs
