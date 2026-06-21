"""AprilTag 位姿服务的 OpenCV 调试覆盖层。

该模块只负责绘制调试画面，不参与位姿计算和串口输出。这样正式采样逻辑
可以在无窗口模式下稳定运行，现场需要观察时再打开 ``--show``。
"""


def build_debug_overlay_items(detections, base_tag_id, tool_tag_id, base_ref_source, last_status):
    """把 AprilTag 检测结果转换为可绘制的覆盖层数据。"""
    tags = []
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id == int(base_tag_id):
            role = "base_ref"
        elif tag_id == int(tool_tag_id):
            role = "tool0"
        else:
            role = "tag"
        tags.append(
            {
                "id": tag_id,
                "label": "ID {} {}".format(tag_id, role),
                "corners": _corners_as_int_pairs(detection.corners),
            }
        )
    return {
        "tags": tags,
        "status_lines": [
            "base_ref_source: {}".format(base_ref_source),
            "last_status: {}".format(last_status),
        ],
    }


def draw_debug_overlay(image_bgr, detections, base_tag_id, tool_tag_id, base_ref_source, last_status):
    """在 OpenCV BGR 图像上绘制 tag 外框、ID 和服务状态。"""
    import cv2

    overlay = build_debug_overlay_items(
        detections=detections,
        base_tag_id=base_tag_id,
        tool_tag_id=tool_tag_id,
        base_ref_source=base_ref_source,
        last_status=last_status,
    )
    for tag in overlay["tags"]:
        corners = tag["corners"]
        color = _tag_color(tag["id"], base_tag_id, tool_tag_id)
        for index in range(4):
            cv2.line(image_bgr, corners[index], corners[(index + 1) % 4], color, 2)
        cv2.putText(
            image_bgr,
            tag["label"],
            corners[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    y = 24
    for line in overlay["status_lines"]:
        cv2.putText(
            image_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return image_bgr


def resize_debug_image(image_bgr, width, height, cv2_module):
    """按指定窗口尺寸缩放调试图像。

    只指定宽或高时保持原始宽高比，避免现场显示被拉伸误导判断。
    """
    target_width = int(width or 0)
    target_height = int(height or 0)
    if target_width <= 0 and target_height <= 0:
        return image_bgr

    image_height, image_width = image_bgr.shape[:2]
    if target_width <= 0:
        target_width = int(round(float(image_width) * float(target_height) / float(image_height)))
    if target_height <= 0:
        target_height = int(round(float(image_height) * float(target_width) / float(image_width)))
    return cv2_module.resize(image_bgr, (target_width, target_height))


def should_quit_from_key(key_code):
    """判断 OpenCV 按键值是否表示退出。"""
    key = int(key_code) & 0xFF
    return key in (ord("q"), 27)


def _corners_as_int_pairs(corners):
    """将检测器输出的浮点角点转换为 OpenCV 可绘制的整数像素坐标。"""
    return [(int(round(point[0])), int(round(point[1]))) for point in corners]


def _tag_color(tag_id, base_tag_id, tool_tag_id):
    """按 tag 角色选择调试颜色。"""
    if int(tag_id) == int(base_tag_id):
        return (0, 255, 0)
    if int(tag_id) == int(tool_tag_id):
        return (255, 0, 0)
    return (255, 255, 255)
