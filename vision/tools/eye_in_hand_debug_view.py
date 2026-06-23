"""Eye-in-hand 单帧与实时调试画面叠加绘制工具。

本模块只负责把感知结果画到 BGR 图像上，不直接读取相机、不保存文件。
这样采集脚本、离线回放脚本和单元测试都可以复用同一套可视化规则。
"""


WINDOW_NAME = "Eye-in-hand Object Debug"


def draw_eye_in_hand_debug_overlay(
    color_bgr,
    result,
    camera_keepout_roi=None,
    cv2_module=None,
):
    """绘制候选物、抓取建议和相机视野禁区。

    颜色约定：
    - 红色：相机视野禁区，表示夹爪或末端结构容易遮挡画面的区域。
    - 绿色：OBJ 候选物，只表示图像候选区域，不表示物体真实形状。
    - 蓝色：GRASP 抓取建议，表示控制侧优先消费的抓取目标。
    - 黄色：有 OBJ 但没有 GRASP，通常表示视野或几何约束未通过。
    """

    cv2 = cv2_module or _import_cv2()
    overlay = color_bgr.copy()
    candidates = list(result.get("candidates", []))
    grasps = list(result.get("grasps", []))
    grasp_ids = set(_as_int(grasp.get("id")) for grasp in grasps)

    if camera_keepout_roi is not None:
        _draw_rect(cv2, overlay, camera_keepout_roi, (0, 0, 255), 2)
        _draw_text(cv2, overlay, "CAMERA KEEP OUT", _label_origin(camera_keepout_roi), (0, 0, 255))

    for candidate in candidates:
        bbox = candidate.get("bbox_pixel")
        if bbox is None:
            continue
        candidate_id = _as_int(candidate.get("id"))
        is_rejected = candidate_id not in grasp_ids
        color = (0, 255, 0) if not is_rejected else (0, 255, 255)
        _draw_rect(cv2, overlay, bbox, color, 2)
        label = "OBJ {0} {1:.2f}".format(candidate_id, float(candidate.get("score", 0.0)))
        if is_rejected:
            label = label + " NO_GRASP"
        _draw_text(cv2, overlay, label, _label_origin(bbox), color)

    for grasp in grasps:
        pixel = grasp_pixel_from_candidate_roi(grasp, candidates)
        if pixel is None:
            continue
        cv2.circle(overlay, pixel, 6, (255, 0, 0), 2)
        label = "GRASP {0} q={1:.2f} v={2:.2f}".format(
            _as_int(grasp.get("id")),
            float(grasp.get("quality", 0.0)),
            float(grasp.get("visibility", 0.0)),
        )
        _draw_text(cv2, overlay, label, (pixel[0] + 8, pixel[1] + 4), (255, 0, 0))

    summary = "OBJ={0} GRASP={1} REJECT={2}".format(
        int(result.get("candidate_count", len(candidates))),
        int(result.get("grasp_count", len(grasps))),
        int(result.get("rejected_grasp_count", max(0, len(candidates) - len(grasps)))),
    )
    _draw_text(cv2, overlay, summary, (10, 24), (255, 255, 255))
    return overlay


def grasp_pixel_from_candidate_roi(grasp, candidates):
    """用同 id 候选物的 ROI 中心作为第一版抓取点显示位置。

    当前 GRASP 协议发送的是 base 坐标系抓取点，现场显示需要落回图像坐标。
    在还没有做 base 点反投影到像素的精确链路前，使用对应 OBJ 的 ROI 中心
    作为调试显示位置，能让操作者直观看到该抓取建议来自哪个候选物。
    """

    grasp_id = _as_int(grasp.get("id"))
    for candidate in candidates:
        if _as_int(candidate.get("id")) != grasp_id:
            continue
        bbox = candidate.get("bbox_pixel")
        if bbox is None:
            return None
        u1, v1, u2, v2 = _normalize_roi(bbox)
        return ((u1 + u2) // 2, (v1 + v2) // 2)
    return None


def should_quit_from_key(key):
    """判断 OpenCV 按键是否表示退出实时窗口。"""

    return key in (ord("q"), ord("Q"), 27)


def _draw_rect(cv2, image, roi, color, thickness):
    """绘制经过归一化的 ROI，避免用户传入反向坐标导致画框异常。"""

    u1, v1, u2, v2 = _normalize_roi(roi)
    cv2.rectangle(image, (u1, v1), (u2, v2), color, thickness)


def _draw_text(cv2, image, text, origin, color):
    """统一文本样式，保证现场调试画面信息密度稳定。"""

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _label_origin(roi):
    """把标签放在 ROI 左上角上方，尽量不遮住目标区域。"""

    u1, v1, _u2, _v2 = _normalize_roi(roi)
    return (u1, max(14, v1 - 6))


def _normalize_roi(roi):
    """归一化 ``u1,v1,u2,v2``，并转换为整数像素坐标。"""

    u1, v1, u2, v2 = [int(value) for value in roi]
    left = min(u1, u2)
    right = max(u1, u2)
    top = min(v1, v2)
    bottom = max(v1, v2)
    return left, top, right, bottom


def _as_int(value):
    """候选 id 来自 JSON 或字典，统一转成整数便于匹配。"""

    return int(value if value is not None else -1)


def _import_cv2():
    """延迟导入 OpenCV，避免无 GUI 环境下导入普通计算模块失败。"""

    import cv2

    return cv2
