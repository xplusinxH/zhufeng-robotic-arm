"""桌面平面拟合与 Z 轴偏差诊断。

本模块只做 eye-in-hand 调试链路中的视觉反验：根据当前深度图估计桌面在
机械臂 base 坐标系下的位置与倾斜程度。它不能替代控制端发送的
``T_base_tool``，但可以反验其 Z 轴偏差，并在明确开启时对候选物 Z 值做临时补偿。
"""

import math

import numpy as np

from coordinate.frame_transform import transform_point
from coordinate.pixel_to_3d import pixel_depth_to_camera


def estimate_table_plane_diagnostics(
    depth_m,
    intrinsics,
    base_from_camera,
    stride=8,
    min_points=80,
    max_fit_rmse_m=0.01,
):
    """拟合当前深度图中的桌面平面，并输出 base 坐标系下的 Z 轴诊断量。

    返回字段说明：
    - ``table_z_offset_m``：拟合桌面在采样区域中心处相对 ``Z_base=0`` 的偏差。
    - ``z_compensation_m``：建议施加到视觉候选 Z 值上的补偿量，等于负偏差。
    - ``table_tilt_deg``：拟合平面法向与 base 坐标系 Z 轴之间的夹角。
    - ``fit_rmse_m``：拟合残差均方根，越小说明采样点越像一个平面。
    """

    points = _sample_base_points(depth_m, intrinsics, base_from_camera, stride=stride)
    if len(points) < int(min_points):
        return _invalid_result("not_enough_points", len(points))

    table_seed_points = _select_likely_table_points(points, min_points)
    plane = _fit_plane_z_from_xy(table_seed_points)
    inlier_points = _select_inlier_points(table_seed_points, plane)
    if len(inlier_points) >= int(min_points):
        plane = _fit_plane_z_from_xy(inlier_points)
        points_for_error = inlier_points
    else:
        points_for_error = table_seed_points

    fit_rmse_m = _plane_rmse(points_for_error, plane)
    z_offset_m = _plane_z_at_mean_xy(points_for_error, plane)
    tilt_deg = _plane_tilt_deg(plane)
    valid = fit_rmse_m <= float(max_fit_rmse_m) or len(points_for_error) >= int(min_points)
    return {
        "valid": bool(valid),
        "reason": "ok" if valid else "fit_error_too_large",
        "sample_count": len(points),
        "inlier_count": len(points_for_error),
        "plane_model": {
            "z_from_xy": {
                "a": plane[0],
                "b": plane[1],
                "c": plane[2],
            }
        },
        "table_z_offset_m": z_offset_m,
        "z_compensation_m": -z_offset_m,
        "table_tilt_deg": tilt_deg,
        "fit_rmse_m": fit_rmse_m,
    }


def apply_table_z_compensation_to_scene(candidates, grasps, table_plane):
    """把桌面 Z 偏差补偿应用到候选物和抓取建议的 base 坐标 Z 分量。

    只修正 Z，不修正 X/Y，也不改写 ``base_from_camera``。每个被补偿的候选物会保留
    ``center_base_raw_m``，方便现场复盘控制端原始位姿链路与视觉补偿后的差异。
    """

    if not table_plane or not table_plane.get("valid"):
        return
    z_compensation_m = float(table_plane.get("z_compensation_m", 0.0))
    if abs(z_compensation_m) <= 0.0:
        return
    for candidate in candidates:
        if "center_base_m" in candidate:
            candidate["center_base_raw_m"] = tuple(candidate["center_base_m"])
            candidate["center_base_m"] = _with_z_offset(
                candidate["center_base_m"],
                z_compensation_m,
            )
            candidate["table_z_compensated"] = True
    for grasp in grasps:
        if "position_base_m" in grasp:
            grasp["position_base_raw_m"] = tuple(grasp["position_base_m"])
            grasp["position_base_m"] = _with_z_offset(
                grasp["position_base_m"],
                z_compensation_m,
            )
            grasp["table_z_compensated"] = True


def _sample_base_points(depth_m, intrinsics, base_from_camera, stride):
    """按固定步长采样深度图，并转换为 base 坐标系点云。"""

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return []
    step = max(1, int(stride))
    points = []
    height, width = depth.shape
    for v in range(0, height, step):
        for u in range(0, width, step):
            depth_value = float(depth[v][u])
            camera_point = pixel_depth_to_camera(u, v, depth_value, intrinsics)
            if camera_point is None:
                continue
            points.append(transform_point(base_from_camera, camera_point))
    return points


def _fit_plane_z_from_xy(points):
    """用最小二乘拟合 ``z = a*x + b*y + c`` 形式的桌面平面。"""

    matrix = np.asarray([[point[0], point[1], 1.0] for point in points], dtype=float)
    values = np.asarray([point[2] for point in points], dtype=float)
    # Jetson 自带的旧版 NumPy 不接受 rcond=None，使用 -1.0 保持旧版默认阈值行为。
    solution, _residuals, _rank, _singular = np.linalg.lstsq(matrix, values, rcond=-1.0)
    return float(solution[0]), float(solution[1]), float(solution[2])


def _select_likely_table_points(points, min_points):
    """优先选择较低的 base-Z 点拟合桌面，减少物体点对平面的污染。"""

    if len(points) <= int(min_points):
        return points
    z_values = np.asarray([point[2] for point in points], dtype=float)
    threshold = float(np.percentile(z_values, 70.0))
    selected = [point for point in points if float(point[2]) <= threshold]
    if len(selected) < int(min_points):
        selected = sorted(points, key=lambda item: float(item[2]))[: int(min_points)]
    return selected


def _select_inlier_points(points, plane):
    """用一次稳健残差过滤降低物体点对桌面拟合的影响。"""

    residuals = np.asarray([abs(_point_plane_residual(point, plane)) for point in points])
    if residuals.size == 0:
        return []
    median = float(np.median(residuals))
    threshold = max(0.006, median * 3.0)
    return [point for point, residual in zip(points, residuals.tolist()) if residual <= threshold]


def _plane_rmse(points, plane):
    """计算点到拟合平面的 Z 向残差均方根。"""

    if not points:
        return 0.0
    errors = [_point_plane_residual(point, plane) for point in points]
    return math.sqrt(sum(error * error for error in errors) / float(len(errors)))


def _plane_z_at_mean_xy(points, plane):
    """计算采样区域中心处的拟合桌面 Z 值，用作 Z_base 偏差。"""

    mean_x = sum(float(point[0]) for point in points) / float(len(points))
    mean_y = sum(float(point[1]) for point in points) / float(len(points))
    a, b, c = plane
    return float(a) * mean_x + float(b) * mean_y + float(c)


def _plane_tilt_deg(plane):
    """计算桌面法向与 base 坐标系 Z 轴的夹角。"""

    a, b, _c = plane
    normal_length = math.sqrt(float(a) * float(a) + float(b) * float(b) + 1.0)
    cos_angle = max(-1.0, min(1.0, 1.0 / normal_length))
    return math.degrees(math.acos(cos_angle))


def _point_plane_residual(point, plane):
    """计算点相对 ``z = a*x + b*y + c`` 的 Z 向残差。"""

    a, b, c = plane
    return float(point[2]) - (float(a) * float(point[0]) + float(b) * float(point[1]) + float(c))


def _with_z_offset(point, z_offset_m):
    """返回只修改 Z 分量后的三维点。"""

    return (float(point[0]), float(point[1]), float(point[2]) + float(z_offset_m))


def _invalid_result(reason, sample_count):
    """构造无效诊断结果，保证 JSON 字段结构稳定。"""

    return {
        "valid": False,
        "reason": reason,
        "sample_count": int(sample_count),
        "inlier_count": 0,
        "plane_model": None,
        "table_z_offset_m": None,
        "z_compensation_m": 0.0,
        "table_tilt_deg": None,
        "fit_rmse_m": None,
    }
