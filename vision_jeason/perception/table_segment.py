"""基于机械臂 base 坐标系高度的桌面物体点提取。

当前项目是 eye-in-hand 相机：D435 安装在机械臂末端执行器上，桌面平面
不再固定在相机坐标系下。正确链路是先把深度点转换到机械臂 base 坐标系，
再使用结构先验 ``Z_base = 0`` 作为桌面平面。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def extract_points_above_base_plane(
    depth_m: Sequence[Sequence[float]],
    intrinsics: Dict[str, float],
    base_from_camera: Sequence[Sequence[float]],
    min_z_base_m: float = 0.01,
    max_z_base_m: float = 0.30,
    stride: int = 1,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> List[Dict[str, object]]:
    """提取 base 坐标系下位于桌面上方的深度点。

    这里直接使用 NumPy 批量计算整张图或 ROI 内的像素高度，避免 Python
    双层循环逐像素计算。Jetson Nano 上按 ``D`` 检测时，慢点主要就在这一步。

    返回字段：
    - ``pixel``：彩色图像素坐标。
    - ``camera_point_m``：相机坐标系三维点，单位米。
    - ``base_point_m``：机械臂 base 坐标系三维点，单位米。
    - ``height_above_table_m``：相对桌面高度；由于桌面为 ``Z_base = 0``，
      该值等于 ``base_point_m[2]``。
    """

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return []

    step = max(1, int(stride))
    min_u, min_v, max_u, max_v = _roi_bounds(depth, roi)
    if max_u < min_u or max_v < min_v:
        return []

    sampled_depth = depth[min_v : max_v + 1 : step, min_u : max_u + 1 : step]
    u_values = np.arange(min_u, max_u + 1, step, dtype=float)
    v_values = np.arange(min_v, max_v + 1, step, dtype=float)
    u_grid, v_grid = np.meshgrid(u_values, v_values)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    z_camera = sampled_depth
    valid_depth = z_camera > 0.0
    x_camera = (u_grid - cx) * z_camera / fx
    y_camera = (v_grid - cy) * z_camera / fy

    transform = np.asarray(base_from_camera, dtype=float)
    x_base = (
        transform[0, 0] * x_camera
        + transform[0, 1] * y_camera
        + transform[0, 2] * z_camera
        + transform[0, 3]
    )
    y_base = (
        transform[1, 0] * x_camera
        + transform[1, 1] * y_camera
        + transform[1, 2] * z_camera
        + transform[1, 3]
    )
    z_base = (
        transform[2, 0] * x_camera
        + transform[2, 1] * y_camera
        + transform[2, 2] * z_camera
        + transform[2, 3]
    )

    height_mask = (
        valid_depth
        & (z_base >= float(min_z_base_m))
        & (z_base <= float(max_z_base_m))
    )
    selected_v, selected_u = np.nonzero(height_mask)

    results = []
    for row_index, col_index in zip(selected_v.tolist(), selected_u.tolist()):
        u = int(u_grid[row_index, col_index])
        v = int(v_grid[row_index, col_index])
        camera_point = (
            float(x_camera[row_index, col_index]),
            float(y_camera[row_index, col_index]),
            float(z_camera[row_index, col_index]),
        )
        base_point = (
            float(x_base[row_index, col_index]),
            float(y_base[row_index, col_index]),
            float(z_base[row_index, col_index]),
        )
        results.append(
            {
                "pixel": (u, v),
                "camera_point_m": camera_point,
                "base_point_m": base_point,
                "height_above_table_m": base_point[2],
            }
        )
    return results


def _roi_bounds(depth_m, roi):
    """计算 ROI 边界，并裁剪到深度图有效范围内。"""

    height = int(depth_m.shape[0])
    width = int(depth_m.shape[1]) if height else 0
    if width == 0:
        return 0, 0, -1, -1
    if roi is None:
        return 0, 0, width - 1, height - 1
    min_u, min_v, max_u, max_v = roi
    return (
        max(0, int(min_u)),
        max(0, int(min_v)),
        min(width - 1, int(max_u)),
        min(height - 1, int(max_v)),
    )
