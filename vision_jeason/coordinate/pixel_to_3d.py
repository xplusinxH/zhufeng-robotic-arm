"""Convert aligned pixel and depth values to camera coordinates."""

from typing import Dict, Optional, Tuple


def pixel_to_camera(
    u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float
) -> Tuple[float, float, float]:
    """Convert a pixel coordinate and depth to camera-frame XYZ in meters."""
    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy
    return x_m, y_m, z_m


def pixel_depth_to_camera(
    u: float, v: float, depth_m: float, intrinsics: Dict[str, float]
) -> Optional[Tuple[float, float, float]]:
    """将对齐像素和深度转换为相机坐标；无效深度返回 None。"""
    if depth_m <= 0:
        return None
    return pixel_to_camera(
        u,
        v,
        depth_m,
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
