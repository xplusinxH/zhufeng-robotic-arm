"""桌面物体候选点聚类。

输入来自 ``table_segment.extract_points_above_base_plane`` 的候选点列表。
当前实现使用像素空间连通性做轻量聚类，适合 Jetson Nano 第一阶段现场验证。
"""

import math
from typing import Dict, List, Sequence


def cluster_candidate_points(
    points: Sequence[Dict[str, object]],
    pixel_radius: int = 3,
    min_points: int = 20,
) -> List[Dict[str, object]]:
    """按像素连通性把桌面上方点聚成物体候选区域。

    返回字段：
    - ``bbox_pixel``：像素包围盒 ``(min_u, min_v, max_u, max_v)``。
    - ``center_pixel``：像素中心。
    - ``center_camera_m``：相机坐标系中心点，单位米。
    - ``center_base_m``：机械臂 base 坐标系中心点，单位米。
    - ``size_m``：候选点云在 XYZ 方向的范围，单位米。
    """

    visited = set()
    clusters = []
    radius = max(1, int(pixel_radius))
    pixel_index = _build_pixel_index(points)

    for index in range(len(points)):
        if index in visited:
            continue
        component_indexes = _collect_component(index, points, radius, visited, pixel_index)
        if len(component_indexes) < min_points:
            continue
        component = [points[item] for item in component_indexes]
        clusters.append(_candidate_from_points(component))

    clusters.sort(key=lambda item: (item["bbox_pixel"][1], item["bbox_pixel"][0]))
    return clusters


def _build_pixel_index(points):
    """建立像素到点索引的哈希表，避免聚类时反复全量扫描。"""

    pixel_index = {}
    for index, point in enumerate(points):
        pixel_index.setdefault(point["pixel"], []).append(index)
    return pixel_index


def _collect_component(start_index, points, radius, visited, pixel_index):
    """从一个点开始收集连通分量。

    旧实现会让每个点扫描全部候选点，复杂度接近 ``O(N^2)``。真实画面里候选点
    一多，Jetson 按 ``D`` 后就容易长时间无响应。这里改为只查当前像素邻域内
    实际存在的点，复杂度接近线性。
    """

    component = []
    queue = [start_index]
    visited.add(start_index)
    while queue:
        current = queue.pop()
        component.append(current)
        current_pixel = points[current]["pixel"]
        for candidate in _neighbor_indexes(current_pixel, radius, pixel_index):
            if candidate in visited:
                continue
            visited.add(candidate)
            queue.append(candidate)
    return component


def _neighbor_indexes(pixel, radius, pixel_index):
    """返回指定像素方形邻域内存在的点索引。"""

    u, v = pixel
    for neighbor_v in range(v - radius, v + radius + 1):
        for neighbor_u in range(u - radius, u + radius + 1):
            indexes = pixel_index.get((neighbor_u, neighbor_v))
            if indexes:
                for index in indexes:
                    yield index


def _candidate_from_points(points):
    """把一个连通点集汇总为物体候选区域描述。"""

    pixels = [point["pixel"] for point in points]
    camera_points = [point["camera_point_m"] for point in points]
    base_points = [
        point.get("base_point_m")
        for point in points
        if point.get("base_point_m") is not None
    ]
    min_u = min(pixel[0] for pixel in pixels)
    max_u = max(pixel[0] for pixel in pixels)
    min_v = min(pixel[1] for pixel in pixels)
    max_v = max(pixel[1] for pixel in pixels)
    xs = [point[0] for point in camera_points]
    ys = [point[1] for point in camera_points]
    zs = [point[2] for point in camera_points]
    candidate = {
        "bbox_pixel": (min_u, min_v, max_u, max_v),
        "center_pixel": (
            _mean([pixel[0] for pixel in pixels]),
            _mean([pixel[1] for pixel in pixels]),
        ),
        "center_camera_m": (_mean(xs), _mean(ys), _mean(zs)),
        "size_m": (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)),
        "point_count": len(points),
    }
    candidate["shape_pixel"] = _build_pixel_shape_summary(pixels)
    candidate["shape_3d_m"] = _build_3d_shape_summary(candidate["size_m"])
    if base_points:
        candidate["center_base_m"] = (
            _mean([point[0] for point in base_points]),
            _mean([point[1] for point in base_points]),
            _mean([point[2] for point in base_points]),
        )
    return candidate


def _mean(values):
    """计算浮点均值。"""

    return sum(float(value) for value in values) / float(len(values))


def _build_pixel_shape_summary(pixels):
    """从候选点集提取像素形状摘要，避免后续只依赖外接矩形理解目标。"""

    min_u = min(pixel[0] for pixel in pixels)
    max_u = max(pixel[0] for pixel in pixels)
    min_v = min(pixel[1] for pixel in pixels)
    max_v = max(pixel[1] for pixel in pixels)
    bbox_width = max_u - min_u + 1
    bbox_height = max_v - min_v + 1
    bbox_area = max(1, bbox_width * bbox_height)
    axis = _principal_axis_from_pixels(pixels)
    return {
        "point_count": len(pixels),
        "bbox_area_pixel": bbox_area,
        "fill_ratio": float(len(pixels)) / float(bbox_area),
        "principal_axis_pixel": axis["principal_axis_pixel"],
        "principal_axis_angle_deg": axis["principal_axis_angle_deg"],
        "major_axis_pixel": axis["major_axis_pixel"],
        "minor_axis_pixel": axis["minor_axis_pixel"],
    }


def _principal_axis_from_pixels(pixels):
    """用二维点集协方差估计主方向，给非矩形物体一个更稳定的方向描述。"""

    mean_u = _mean([pixel[0] for pixel in pixels])
    mean_v = _mean([pixel[1] for pixel in pixels])
    if len(pixels) <= 1:
        return {
            "principal_axis_pixel": (1.0, 0.0),
            "principal_axis_angle_deg": 0.0,
            "major_axis_pixel": 0.0,
            "minor_axis_pixel": 0.0,
        }

    uu = _mean([(pixel[0] - mean_u) * (pixel[0] - mean_u) for pixel in pixels])
    vv = _mean([(pixel[1] - mean_v) * (pixel[1] - mean_v) for pixel in pixels])
    uv = _mean([(pixel[0] - mean_u) * (pixel[1] - mean_v) for pixel in pixels])
    angle = 0.5 * math.atan2(2.0 * uv, uu - vv)
    trace = uu + vv
    delta = math.sqrt(max(0.0, (uu - vv) * (uu - vv) + 4.0 * uv * uv))
    major_variance = max(0.0, 0.5 * (trace + delta))
    minor_variance = max(0.0, 0.5 * (trace - delta))
    return {
        "principal_axis_pixel": (math.cos(angle), math.sin(angle)),
        "principal_axis_angle_deg": math.degrees(angle),
        "major_axis_pixel": 4.0 * math.sqrt(major_variance),
        "minor_axis_pixel": 4.0 * math.sqrt(minor_variance),
    }


def _build_3d_shape_summary(size_m):
    """把三维包围尺寸整理为长轴、短轴和高度，供抓取规划使用。"""

    x_size, y_size, z_size = [float(value) for value in size_m]
    planar_sizes = sorted((abs(x_size), abs(y_size)), reverse=True)
    return {
        "major_axis_m": planar_sizes[0],
        "minor_axis_m": planar_sizes[1],
        "height_m": abs(z_size),
    }
