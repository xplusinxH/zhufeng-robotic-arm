"""像素坐标与深度值转换为相机三维坐标。

前提：深度图已经对齐到彩色图，因此输入像素 ``u, v`` 使用彩色图坐标。
输出坐标遵循 D435 相机坐标系，单位米。
"""

from typing import Dict, Optional, Tuple


def pixel_to_camera(
    u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float
) -> Tuple[float, float, float]:
    """按针孔模型把单个像素点转换为相机坐标。

    ``fx/fy/cx/cy`` 单位为像素，``z_m`` 单位为米。
    """
    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy
    return x_m, y_m, z_m


def pixel_depth_to_camera(
    u: float, v: float, depth_m: float, intrinsics: Dict[str, float]
) -> Optional[Tuple[float, float, float]]:
    """将对齐像素和深度转换为相机坐标。

    深度小于等于 0 表示 D435 未得到有效测距，返回 ``None``，避免后续
    误把无效深度当成相机前方原点附近的真实目标。
    """
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
