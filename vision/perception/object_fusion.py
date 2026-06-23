"""基于机械臂基坐标系高度的未知物体候选生成。

当前工程采用 eye-in-hand 相机结构：D435 安装在机械臂末端，运行时由
控制侧通过串口发送 ``T_base_tool``，手眼标定提供固定的
``T_tool_camera``，因此视觉侧最终使用 ``T_base_camera`` 把深度点转换到
机械臂基坐标系下判断。桌面平面不再在相机坐标系中单独标定，而是使用
机械结构先验：桌面对应基坐标系 ``Z_base = 0``。
"""

from perception.object_cluster import cluster_candidate_points
from perception.table_plane import estimate_table_plane_diagnostics
from perception.table_segment import extract_points_above_base_plane
from coordinate.frame_transform import transform_point
from coordinate.pixel_to_3d import pixel_depth_to_camera

import numpy as np


def build_base_height_object_candidates(
    depth_m,
    intrinsics,
    base_from_camera,
    min_points=20,
    pixel_radius=3,
    stride=1,
    min_z_base_m=0.01,
    max_z_base_m=0.30,
):
    """从深度图中提取桌面上方的未知物体候选。

    参数中的 ``base_from_camera`` 必须是当前帧对应的实时外参，也就是
    ``T_base_camera = T_base_tool * T_tool_camera``。这样即使相机随末端运动，
    高度过滤仍然在稳定的机械臂基坐标系中完成。
    """

    points = extract_points_above_base_plane(
        depth_m,
        intrinsics,
        base_from_camera,
        min_z_base_m=min_z_base_m,
        max_z_base_m=max_z_base_m,
        stride=stride,
    )
    clusters = cluster_candidate_points(points, pixel_radius=pixel_radius, min_points=min_points)
    return [_candidate_from_cluster(index, cluster) for index, cluster in enumerate(clusters)]


def build_depth_foreground_object_candidates(
    depth_m,
    intrinsics,
    base_from_camera,
    min_points=20,
    pixel_radius=3,
    stride=1,
    foreground_delta_m=0.02,
    background_percentile=70.0,
):
    """用相对深度提取当前画面中的前景物体候选。

    当前真实 ``T_base_tool`` 和 ``T_tool_camera`` 还没接入，单靠 base 绝对高度会把
    大面积桌面误当成一个物体。该调试模式先估计当前画面的桌面/背景深度，
    再保留比背景更靠近相机的点，用于现场快速观察候选框是否合理。
    """

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return []

    step = max(1, int(stride))
    sampled = depth[::step, ::step]
    valid_depth = sampled[sampled > 0.0]
    if valid_depth.size == 0:
        return []

    background_depth = float(np.percentile(valid_depth, float(background_percentile)))
    foreground_limit = background_depth - float(foreground_delta_m)
    mask = (sampled > 0.0) & (sampled <= foreground_limit)
    selected_v, selected_u = np.nonzero(mask)

    points = []
    for row_index, col_index in zip(selected_v.tolist(), selected_u.tolist()):
        u = int(col_index * step)
        v = int(row_index * step)
        camera_point = pixel_depth_to_camera(u, v, float(depth[v][u]), intrinsics)
        if camera_point is None:
            continue
        base_point = transform_point(base_from_camera, camera_point)
        points.append(
            {
                "pixel": (u, v),
                "camera_point_m": camera_point,
                "base_point_m": base_point,
                "height_above_table_m": base_point[2],
            }
        )

    clusters = cluster_candidate_points(points, pixel_radius=pixel_radius, min_points=min_points)
    image_size = (int(depth.shape[1]), int(depth.shape[0]))
    candidates = [_candidate_from_cluster(index, cluster) for index, cluster in enumerate(clusters)]
    for candidate in candidates:
        candidate["source"] = "depth_foreground"
        candidate["background_depth_m"] = background_depth
        candidate["foreground_delta_m"] = float(foreground_delta_m)
    return _filter_and_reindex_depth_foreground_candidates(candidates, image_size)


def build_table_plane_object_candidates(
    depth_m,
    intrinsics,
    base_from_camera,
    min_points=20,
    pixel_radius=3,
    stride=1,
    min_height_above_plane_m=0.006,
    max_height_above_plane_m=0.30,
    table_plane_min_points=80,
):
    """基于拟合桌面平面提取物体候选。

    该模式比 ``depth_foreground`` 更适合斜视桌面：先在 base 坐标系拟合桌面平面，
    再计算每个深度点高出该平面的距离。只要物体高出桌面超过阈值，即使它和背景
    的相机深度差不足 2cm，也可以被保留下来。
    """

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return []
    table_plane = estimate_table_plane_diagnostics(
        depth_m=depth,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        stride=stride,
        min_points=table_plane_min_points,
    )
    if not table_plane.get("valid"):
        return []
    plane_model = table_plane["plane_model"]["z_from_xy"]
    points = _collect_points_above_table_plane(
        depth,
        intrinsics,
        base_from_camera,
        plane_model,
        stride=stride,
        min_height_above_plane_m=min_height_above_plane_m,
        max_height_above_plane_m=max_height_above_plane_m,
    )
    clusters = cluster_candidate_points(points, pixel_radius=pixel_radius, min_points=min_points)
    image_size = (int(depth.shape[1]), int(depth.shape[0]))
    candidates = [_candidate_from_cluster(index, cluster) for index, cluster in enumerate(clusters)]
    for candidate in candidates:
        candidate["source"] = "table_plane"
        candidate["table_plane"] = table_plane
        candidate["min_height_above_plane_m"] = float(min_height_above_plane_m)
        candidate["max_height_above_plane_m"] = float(max_height_above_plane_m)
    return _filter_and_reindex_depth_foreground_candidates(candidates, image_size)


def _collect_points_above_table_plane(
    depth,
    intrinsics,
    base_from_camera,
    plane_model,
    stride,
    min_height_above_plane_m,
    max_height_above_plane_m,
):
    """收集高出拟合桌面平面的深度点，作为后续连通域聚类输入。"""

    step = max(1, int(stride))
    points = []
    height, width = depth.shape
    for v in range(0, height, step):
        for u in range(0, width, step):
            camera_point = pixel_depth_to_camera(u, v, float(depth[v][u]), intrinsics)
            if camera_point is None:
                continue
            base_point = transform_point(base_from_camera, camera_point)
            height_above_plane = _height_above_table_plane(base_point, plane_model)
            plane_distance = abs(height_above_plane)
            if plane_distance < float(min_height_above_plane_m):
                continue
            if plane_distance > float(max_height_above_plane_m):
                continue
            points.append(
                {
                    "pixel": (u, v),
                    "camera_point_m": camera_point,
                    "base_point_m": base_point,
                    "height_above_table_m": height_above_plane,
                    "table_plane_distance_m": plane_distance,
                }
            )
    return points


def _height_above_table_plane(base_point, plane_model):
    """计算 base 坐标点相对拟合桌面平面的 Z 向高度。"""

    table_z = (
        float(plane_model["a"]) * float(base_point[0])
        + float(plane_model["b"]) * float(base_point[1])
        + float(plane_model["c"])
    )
    return float(base_point[2]) - table_z


def _candidate_from_cluster(candidate_id, cluster):
    """把几何聚类结果整理成后续抓取规划可消费的候选结构。"""

    candidate = dict(cluster)
    candidate["id"] = int(candidate_id)
    candidate["class_name"] = "unknown"
    candidate["score"] = 1.0
    candidate["source"] = "base_height"
    return candidate


def _filter_and_reindex_depth_foreground_candidates(candidates, image_size):
    """过滤明显不像可抓取物体的深度前景块，并重新分配连续候选 id。

    深度前景模式用于控制侧位姿尚未完整接入时的现场调试。斜视或桌面边缘会产生
    贴边大块、底部薄条等误检，这些区域即使被深度阈值选中，也不应该进入 GRASP。
    """

    filtered = []
    for candidate in candidates:
        if _is_rejected_depth_foreground_candidate(candidate, image_size):
            continue
        candidate["id"] = len(filtered)
        filtered.append(candidate)
    return filtered


def _is_rejected_depth_foreground_candidate(candidate, image_size):
    """判断候选块是否更像桌面边缘或画面边界误检，而不是独立物体。"""

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return False
    if min(image_width, image_height) < 10:
        return False

    u1, v1, u2, v2 = [int(value) for value in candidate["bbox_pixel"]]
    bbox_width = max(1, u2 - u1 + 1)
    bbox_height = max(1, v2 - v1 + 1)
    area_ratio = float(bbox_width * bbox_height) / float(image_width * image_height)
    width_ratio = float(bbox_width) / float(image_width)
    height_ratio = float(bbox_height) / float(image_height)
    border_margin = max(2, int(round(min(image_width, image_height) * 0.02)))
    touches_left = u1 <= border_margin
    touches_right = u2 >= image_width - 1 - border_margin
    touches_bottom = v2 >= image_height - 1 - border_margin
    touches_top = v1 <= border_margin
    shape_3d = candidate.get("shape_3d_m", {})
    major_axis_m = float(shape_3d.get("major_axis_m", 0.0))
    height_m = float(shape_3d.get("height_m", 0.0))

    if area_ratio > 0.45:
        return True
    if major_axis_m > 0.25 and (touches_left or touches_right or width_ratio > 0.70):
        return True
    if (touches_left or touches_right) and width_ratio > 0.70 and height_ratio > 0.25:
        return True
    if touches_bottom and height_m < 0.004:
        return True
    if touches_bottom and bbox_height <= max(3, int(image_height * 0.12)) and width_ratio > 0.35:
        return True
    if touches_top and touches_bottom and area_ratio > 0.20:
        return True
    return False
