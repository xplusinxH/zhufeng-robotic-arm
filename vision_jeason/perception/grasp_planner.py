"""相机视野优先的第一版抓取建议生成。

D435 安装在夹爪上方，抓取动作如果遮挡相机视线，会导致末端接近阶段无法
继续确认目标位置。本模块先实现一个保守的二维 ROI 启发式：把夹爪容易遮挡
相机视野的区域建模为 ``camera_keepout_roi``，候选物 ROI 与该区域重叠越多，
``visibility`` 评分越低。低于阈值时不生成 ``GRASP`` 建议。

该模块不是最终抓取规划器；后续可在同一接口下接入夹爪 CAD 投影、点云法向、
碰撞检测和多姿态搜索。
"""


DEFAULT_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


def estimate_visibility_score(candidate_roi, camera_keepout_roi=None):
    """估算候选 ROI 在抓取前保持可见的程度。

    返回值范围为 ``0.0`` 到 ``1.0``。当前启发式按候选 ROI 与相机视野禁区的
    重叠面积计算；完全不重叠为 1，完全被禁区覆盖为 0。
    """

    if camera_keepout_roi is None:
        return 1.0
    candidate_area = _roi_area(candidate_roi)
    if candidate_area <= 0:
        return 0.0
    overlap_area = _roi_area(_intersect_roi(candidate_roi, camera_keepout_roi))
    return _clamp01(1.0 - overlap_area / float(candidate_area))


def build_visibility_aware_grasp(
    candidate,
    image_size,
    camera_keepout_roi=None,
    min_visibility=0.60,
    width_margin_m=0.01,
):
    """从候选物生成第一版视野安全抓取建议。

    ``image_size`` 预留给后续基于图像尺寸自动生成视野禁区；当前版本主要使用
    显式传入的 ``camera_keepout_roi``。若 ``visibility`` 低于阈值，返回
    ``None``，表示该候选不应发送给控制端执行。
    """

    _ = image_size
    visibility = estimate_visibility_score(
        candidate["bbox_pixel"],
        camera_keepout_roi=camera_keepout_roi,
    )
    if visibility < float(min_visibility):
        return None

    center_base_m = _as_float_tuple(candidate["center_base_m"], 3)
    size_m = _as_float_tuple(candidate.get("size_m", (0.0, 0.0, 0.0)), 3)
    shape_3d = _shape_3d_from_candidate(candidate, size_m)
    width_m = shape_3d["minor_axis_m"] + float(width_margin_m)
    base_quality = _clamp01(candidate.get("score", 1.0))
    quality = base_quality * visibility
    return {
        "id": int(candidate.get("id", 0)),
        "position_base_m": center_base_m,
        "orientation_xyzw": DEFAULT_ORIENTATION_XYZW,
        "width_m": width_m,
        "quality": quality,
        "visibility": visibility,
        "approach": "visibility_first_top",
        "object_major_axis_m": shape_3d["major_axis_m"],
        "object_minor_axis_m": shape_3d["minor_axis_m"],
        "object_height_m": shape_3d["height_m"],
    }


def build_visibility_aware_grasps(
    candidates,
    image_size,
    camera_keepout_roi=None,
    min_visibility=0.60,
    width_margin_m=0.01,
):
    """批量生成抓取建议，并按综合质量从高到低排序。"""

    grasps = []
    for candidate in candidates:
        grasp = build_visibility_aware_grasp(
            candidate,
            image_size=image_size,
            camera_keepout_roi=camera_keepout_roi,
            min_visibility=min_visibility,
            width_margin_m=width_margin_m,
        )
        if grasp is not None:
            grasps.append(grasp)
    grasps.sort(key=lambda item: item["quality"], reverse=True)
    return grasps


def _intersect_roi(left, right):
    """计算两个 ROI 的交集，ROI 格式为 ``(u1, v1, u2, v2)``。"""

    left_u1, left_v1, left_u2, left_v2 = _as_float_tuple(left, 4)
    right_u1, right_v1, right_u2, right_v2 = _as_float_tuple(right, 4)
    return (
        max(left_u1, right_u1),
        max(left_v1, right_v1),
        min(left_u2, right_u2),
        min(left_v2, right_v2),
    )


def _roi_area(roi):
    """计算 ROI 面积，退化或反向 ROI 面积为 0。"""

    u1, v1, u2, v2 = _as_float_tuple(roi, 4)
    return max(0.0, u2 - u1) * max(0.0, v2 - v1)


def _as_float_tuple(values, expected_count):
    """把定长序列转换为浮点元组。"""

    if len(values) != expected_count:
        raise ValueError("需要 {} 个数值".format(expected_count))
    return tuple(float(value) for value in values)


def _shape_3d_from_candidate(candidate, size_m):
    """读取候选物的三维形状摘要，缺失时退回到包围盒平面尺寸。"""

    shape_3d = candidate.get("shape_3d_m") or {}
    if shape_3d:
        major_axis_m = float(shape_3d.get("major_axis_m", 0.0))
        minor_axis_m = float(shape_3d.get("minor_axis_m", 0.0))
        height_m = float(shape_3d.get("height_m", 0.0))
    else:
        planar_sizes = sorted((abs(float(size_m[0])), abs(float(size_m[1]))), reverse=True)
        major_axis_m = planar_sizes[0]
        minor_axis_m = planar_sizes[0]
        height_m = abs(float(size_m[2]))
    return {
        "major_axis_m": major_axis_m,
        "minor_axis_m": minor_axis_m,
        "height_m": height_m,
    }


def _clamp01(value):
    """把评分限制在 0 到 1。"""

    value = float(value)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
