"""桌面整理目标与抓取建议输出的 ASCII 串口协议。

正式抓取不能只依赖物体几何中心。视觉侧应先发送物体候选摘要，再发送
可执行的抓取建议。候选物帧中的 ``bbox`` 只是图像 ROI，不表示物体是
长方体；真正给控制端执行的是 ``GRASP`` 帧中的抓取位姿、夹爪开口和质量。
所有位置和长度字段统一使用毫米，帧格式使用 ``@`` 作为帧头、``#`` 作为帧尾。
"""


def format_no_target() -> str:
    """格式化旧版无目标帧，保留给早期调试脚本兼容。"""
    return "@NO_TARGET#"


def format_target(class_name: str, x_mm: float, y_mm: float, z_mm: float, score: float) -> str:
    """格式化旧版单目标帧，保留给早期调试脚本兼容。

    参数：
    - ``class_name``：目标类别，未知物体可传 ``unknown``。
    - ``x_mm/y_mm/z_mm``：工作坐标系或约定输出坐标，单位毫米。
    - ``score``：检测或融合置信度，范围建议为 0 到 1。
    """
    return f"@TARGET,{class_name},{x_mm:.1f},{y_mm:.1f},{z_mm:.1f},{score:.2f}#"


def format_no_object() -> str:
    """格式化新版无候选物帧。"""

    return "@NOOBJ#"


def format_end(count):
    """格式化一组多帧输出的结束帧。"""

    return "@END,{}#".format(int(count))


def format_object_candidate(candidate):
    """格式化物体候选摘要帧。

    ``candidate`` 来自几何候选或识别融合结果。该帧只描述“哪里可能有物体”
    和图像 ROI，不能被控制端解释为完整几何形状。
    """

    candidate_id = int(candidate.get("id", 0))
    class_name = _safe_token(candidate.get("class_name", "unknown"))
    score = _clamp_score(candidate.get("score", 0.0))
    x_m, y_m, z_m = _required_vector(candidate, "center_base_m", 3)
    min_u, min_v, max_u, max_v = _required_vector(candidate, "bbox_pixel", 4, int)
    point_count = int(candidate.get("point_count", 0))
    source = _safe_token(candidate.get("source", "unknown"))
    return (
        "@OBJ,{},{},{:.2f},{:.1f},{:.1f},{:.1f},{},{},{},{},{},{}#".format(
            candidate_id,
            class_name,
            score,
            x_m * 1000.0,
            y_m * 1000.0,
            z_m * 1000.0,
            min_u,
            min_v,
            max_u,
            max_v,
            point_count,
            source,
        )
    )


def format_grasp_candidate(grasp):
    """格式化抓取建议帧。

    ``grasp`` 是视觉侧给控制端的可执行建议，而不是物体形状描述。控制端
    应优先消费该帧中的抓取点、姿态、夹爪开口、质量评分和相机视野安全评分。
    """

    grasp_id = int(grasp.get("id", 0))
    x_m, y_m, z_m = _required_vector(grasp, "position_base_m", 3)
    qx, qy, qz, qw = _required_vector(grasp, "orientation_xyzw", 4)
    width_m = float(grasp.get("width_m", 0.0))
    quality = _clamp_score(grasp.get("quality", 0.0))
    visibility = _clamp_score(grasp.get("visibility", 1.0))
    approach = _safe_token(grasp.get("approach", "unknown"))
    return (
        "@GRASP,{},{:.1f},{:.1f},{:.1f},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f},{:.2f},{:.2f},{}#".format(
            grasp_id,
            x_m * 1000.0,
            y_m * 1000.0,
            z_m * 1000.0,
            qx,
            qy,
            qz,
            qw,
            width_m * 1000.0,
            quality,
            visibility,
            approach,
        )
    )


def format_error(code, message):
    """格式化新版错误帧，并清理会破坏串口帧结构的字符。"""

    return "@ERR,{},{}#".format(_safe_token(code), _safe_message(message))


def _required_vector(data, key, expected_count, item_type=float):
    """读取固定长度向量字段，缺失时给出明确错误。"""

    if key not in data:
        raise ValueError("缺少字段 {}".format(key))
    values = data[key]
    if len(values) != expected_count:
        raise ValueError("{} 需要 {} 个数值".format(key, expected_count))
    return tuple(item_type(value) for value in values)


def _clamp_score(value):
    """把置信度限制在 0 到 1，避免控制端收到异常评分。"""

    score = float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _safe_token(value):
    """清理协议 token，避免逗号和帧头帧尾破坏解析。"""

    text = str(value).strip()
    for unsafe in (",", "#", "@", " ", "\r", "\n", "\t"):
        text = text.replace(unsafe, "_")
    return text or "unknown"


def _safe_message(value):
    """清理错误消息，保留可读性但不破坏帧结构。"""

    text = str(value).strip()
    for unsafe in (",", "#", "@", "\r", "\n", "\t"):
        text = text.replace(unsafe, " ")
    return " ".join(text.split()) or "unknown"
