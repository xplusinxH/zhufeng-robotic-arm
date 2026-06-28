"""YOLO 检测框到深度 3D 几何的高速转换。

Jetson 部署时不再用全图深度聚类作为主识别入口，而是先由 YOLO 给出 RGB
二维检测框，再只在框内读取 D435 对齐深度。这样点云计算规模从整幅图下降到
少量 ROI，能明显降低坐标输出延迟。
"""

from coordinate.frame_transform import transform_point

import numpy as np


def depth_roi_to_object_geometry(
    depth_m,
    bbox_pixel,
    intrinsics,
    min_points=20,
    stride=1,
    depth_percentile=50.0,
):
    """把一个 YOLO 像素框内的深度点转换成物体几何摘要。

    返回值只描述相机坐标系下的几何；如果需要 base 坐标系，调用
    ``build_yolo_depth_candidates`` 传入 ``base_from_camera``。这里使用 NumPy
    一次性处理 ROI 内像素，避免在 Jetson 上逐点 Python 循环拖慢速度。
    """

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return None

    clipped_bbox = _clip_bbox(bbox_pixel, width=depth.shape[1], height=depth.shape[0])
    if clipped_bbox is None:
        return None
    u1, v1, u2, v2 = clipped_bbox
    step = max(1, int(stride))
    roi_depth = depth[v1 : v2 + 1 : step, u1 : u2 + 1 : step]
    valid_mask = np.isfinite(roi_depth) & (roi_depth > 0.0)
    if int(valid_mask.sum()) < int(min_points):
        return None

    local_v, local_u = np.nonzero(valid_mask)
    z_values = roi_depth[valid_mask].astype(float)
    u_values = (u1 + local_u * step).astype(float)
    v_values = (v1 + local_v * step).astype(float)
    camera_points = _project_pixels_to_camera_points(
        u_values,
        v_values,
        z_values,
        intrinsics,
    )
    if camera_points.shape[0] < int(min_points):
        return None

    center_camera_m = _robust_center(camera_points, depth_percentile)
    min_xyz = np.min(camera_points, axis=0)
    max_xyz = np.max(camera_points, axis=0)
    size_m = max_xyz - min_xyz
    major_axis_m, minor_axis_m = _planar_axes(size_m)
    return {
        "bbox_pixel": clipped_bbox,
        "depth_point_count": int(camera_points.shape[0]),
        "center_camera_m": _tuple3(center_camera_m),
        "size_m": _tuple3(size_m),
        "shape_3d_m": {
            "major_axis_m": float(major_axis_m),
            "minor_axis_m": float(minor_axis_m),
            "height_m": float(abs(size_m[2])),
        },
        "depth_percentile": float(depth_percentile),
    }


def build_yolo_depth_candidates(
    detections,
    depth_m,
    intrinsics,
    base_from_camera=None,
    min_points=20,
    stride=1,
):
    """把 YOLO 检测结果批量转换成后续 GRASP/串口协议可消费的候选物。

    ``detections`` 每项至少包含 ``bbox_pixel``，可选 ``class_name`` 和 ``score``。
    无效深度、点数不足或超出画面的框会被跳过。
    """

    candidates = []
    for detection in detections:
        geometry = depth_roi_to_object_geometry(
            depth_m,
            bbox_pixel=detection["bbox_pixel"],
            intrinsics=intrinsics,
            min_points=min_points,
            stride=stride,
        )
        if geometry is None:
            continue
        center_camera_m = geometry["center_camera_m"]
        candidate = {
            "id": len(candidates),
            "class_name": str(detection.get("class_name", "unknown")),
            "score": float(detection.get("score", 1.0)),
            "source": "yolo_depth",
            "bbox_pixel": geometry["bbox_pixel"],
            "point_count": geometry["depth_point_count"],
            "center_camera_m": center_camera_m,
            "size_m": geometry["size_m"],
            "shape_3d_m": geometry["shape_3d_m"],
        }
        if base_from_camera is not None:
            candidate["center_base_m"] = transform_point(base_from_camera, center_camera_m)
        candidates.append(candidate)
    return candidates


def _project_pixels_to_camera_points(u_values, v_values, z_values, intrinsics):
    """使用针孔模型把 ROI 像素批量投影到相机坐标系。"""

    fx = float(_intrinsic_value(intrinsics, "fx"))
    fy = float(_intrinsic_value(intrinsics, "fy"))
    cx = float(_intrinsic_value(intrinsics, "cx"))
    cy = float(_intrinsic_value(intrinsics, "cy"))
    x_values = (u_values - cx) * z_values / fx
    y_values = (v_values - cy) * z_values / fy
    return np.stack((x_values, y_values, z_values), axis=1)


def _robust_center(camera_points, depth_percentile):
    """用中位深度附近的点求中心，降低背景深度混入 ROI 时的影响。"""

    z_values = camera_points[:, 2]
    target_z = float(np.percentile(z_values, float(depth_percentile)))
    tolerance = max(0.003, float(np.std(z_values)) * 0.75)
    selected = camera_points[np.abs(z_values - target_z) <= tolerance]
    if selected.shape[0] == 0:
        selected = camera_points
    return np.median(selected, axis=0)


def _clip_bbox(bbox_pixel, width, height):
    """把检测框裁剪到图像范围内，保持 ``u1,v1,u2,v2`` 为闭区间。"""

    if len(bbox_pixel) != 4:
        raise ValueError("bbox_pixel 必须是 u1,v1,u2,v2 四个数")
    u1, v1, u2, v2 = [int(round(float(value))) for value in bbox_pixel]
    left = max(0, min(u1, u2))
    top = max(0, min(v1, v2))
    right = min(int(width) - 1, max(u1, u2))
    bottom = min(int(height) - 1, max(v1, v2))
    if right < left or bottom < top:
        return None
    return (left, top, right, bottom)


def _planar_axes(size_m):
    """从 3D 包围盒的 X/Y 尺寸估算平面长轴和短轴。"""

    planar = sorted((abs(float(size_m[0])), abs(float(size_m[1]))), reverse=True)
    return planar[0], planar[1]


def _intrinsic_value(intrinsics, key):
    """兼容 dict 和 RealSense intrinsics 对象。"""

    if isinstance(intrinsics, dict):
        return intrinsics[key]
    return getattr(intrinsics, key)


def _tuple3(values):
    """把 NumPy 数组转换成 JSON 友好的三元浮点元组。"""

    return (float(values[0]), float(values[1]), float(values[2]))
