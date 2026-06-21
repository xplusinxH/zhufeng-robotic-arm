"""基于机械臂 base 坐标系高度的桌面物体点提取。

当前项目为 eye-in-hand 相机：D435 装在机械臂执行器上，桌面平面不是固定在
相机坐标系下的平面。正确做法是先把相机点转换到机械臂 base 坐标系，
再使用结构先验 ``Z_base = 0`` 作为桌面平面。
"""

from typing import Dict, List, Optional, Sequence, Tuple

from coordinate.frame_transform import transform_point
from coordinate.pixel_to_3d import pixel_depth_to_camera


def extract_points_above_base_plane(
    depth_m: Sequence[Sequence[float]],
    intrinsics: Dict[str, float],
    base_from_camera: Sequence[Sequence[float]],
    min_z_base_m: float = 0.01,
    max_z_base_m: float = 0.30,
    stride: int = 1,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> List[Dict[str, object]]:
    """提取 base 坐标系下位于桌面上方的点。

    参数：
    - ``depth_m``：已对齐到彩色图的深度图，单位米。
    - ``intrinsics``：彩色图内参字典，包含 ``fx/fy/cx/cy``。
    - ``base_from_camera``：实时 ``T_base_camera``。
    - ``min_z_base_m/max_z_base_m``：保留的桌面上方高度范围，单位米。

    返回：
    - ``pixel``：彩色图像素坐标。
    - ``camera_point_m``：相机坐标系三维点。
    - ``base_point_m``：机械臂 base 坐标系三维点。
    - ``height_above_table_m``：相对桌面高度；由于桌面为 ``Z_base=0``，
      该值等于 ``base_point_m[2]``。
    """
    results = []
    step = max(1, int(stride))
    min_u, min_v, max_u, max_v = _roi_bounds(depth_m, roi)
    for v in range(min_v, max_v + 1, step):
        row = depth_m[v]
        for u in range(min_u, max_u + 1, step):
            camera_point = pixel_depth_to_camera(u, v, float(row[u]), intrinsics)
            if camera_point is None:
                continue
            base_point = transform_point(base_from_camera, camera_point)
            height = base_point[2]
            if min_z_base_m <= height <= max_z_base_m:
                results.append(
                    {
                        "pixel": (u, v),
                        "camera_point_m": camera_point,
                        "base_point_m": base_point,
                        "height_above_table_m": height,
                    }
                )
    return results


def _roi_bounds(depth_m, roi):
    """计算 ROI 边界，并裁剪到深度图有效范围内。"""
    height = len(depth_m)
    width = len(depth_m[0]) if height else 0
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
