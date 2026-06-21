"""桌面物体候选点聚类。

输入来自 ``table_segment.extract_points_above_base_plane`` 的候选点列表。
当前实现使用像素空间连通性做轻量聚类，适合 Jetson Nano 第一阶段验证；
后续如需更强鲁棒性，可在同一接口下替换为 DBSCAN 或点云聚类。
"""

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

    for index in range(len(points)):
        if index in visited:
            continue
        component_indexes = _collect_component(index, points, radius, visited)
        if len(component_indexes) < min_points:
            continue
        component = [points[item] for item in component_indexes]
        clusters.append(_candidate_from_points(component))

    clusters.sort(key=lambda item: (item["bbox_pixel"][1], item["bbox_pixel"][0]))
    return clusters


def _collect_component(start_index, points, radius, visited):
    """从一个点开始做广度优先搜索，收集同一连通分量。"""
    component = []
    queue = [start_index]
    visited.add(start_index)
    while queue:
        current = queue.pop(0)
        component.append(current)
        current_pixel = points[current]["pixel"]
        for candidate in range(len(points)):
            if candidate in visited:
                continue
            if _pixels_connected(current_pixel, points[candidate]["pixel"], radius):
                visited.add(candidate)
                queue.append(candidate)
    return component


def _pixels_connected(left, right, radius):
    """判断两个像素点是否在方形邻域内连通。"""
    return max(abs(left[0] - right[0]), abs(left[1] - right[1])) <= radius


def _candidate_from_points(points):
    """把一个连通点集汇总为物体候选区域描述。"""
    pixels = [point["pixel"] for point in points]
    camera_points = [point["camera_point_m"] for point in points]
    base_points = [point.get("base_point_m") for point in points if point.get("base_point_m") is not None]
    min_u = min(pixel[0] for pixel in pixels)
    max_u = max(pixel[0] for pixel in pixels)
    min_v = min(pixel[1] for pixel in pixels)
    max_v = max(pixel[1] for pixel in pixels)
    xs = [point[0] for point in camera_points]
    ys = [point[1] for point in camera_points]
    zs = [point[2] for point in camera_points]
    candidate = {
        "bbox_pixel": (min_u, min_v, max_u, max_v),
        "center_pixel": (_mean([pixel[0] for pixel in pixels]), _mean([pixel[1] for pixel in pixels])),
        "center_camera_m": (_mean(xs), _mean(ys), _mean(zs)),
        "size_m": (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)),
        "point_count": len(points),
    }
    if base_points:
        candidate["center_base_m"] = (
            _mean([point[0] for point in base_points]),
            _mean([point[1] for point in base_points]),
            _mean([point[2] for point in base_points]),
        )
    return candidate


def _mean(values):
    return sum(float(value) for value in values) / float(len(values))
