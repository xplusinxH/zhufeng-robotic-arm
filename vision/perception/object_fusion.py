"""基于机械臂基坐标系高度的未知物体候选生成。

当前工程采用 eye-in-hand 相机结构：D435 安装在机械臂末端，运行时由
控制侧通过串口发送 ``T_base_tool``，手眼标定提供固定的
``T_tool_camera``，因此视觉侧最终使用 ``T_base_camera`` 把深度点转换到
机械臂基坐标系下判断。桌面平面不再在相机坐标系中单独标定，而是使用
机械结构先验：桌面对应基坐标系 ``Z_base = 0``。
"""

from perception.object_cluster import cluster_candidate_points
from perception.table_segment import extract_points_above_base_plane


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


def _candidate_from_cluster(candidate_id, cluster):
    """把几何聚类结果整理成后续抓取规划可消费的候选结构。"""

    candidate = dict(cluster)
    candidate["id"] = int(candidate_id)
    candidate["class_name"] = "unknown"
    candidate["score"] = 1.0
    candidate["source"] = "base_height"
    return candidate
