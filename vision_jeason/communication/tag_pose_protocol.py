"""AprilTag 末端 6D 位姿串口协议。

主协议命令：
- 电脑端发送 ``@GET_TAG_POSE#``。
- Jetson 采样多帧 AprilTag，返回一行 ``sukinee_tag_pose_v1`` JSON。

协议输出的 ``position_m`` 使用米，``orientation_xyzw`` 使用 XYZW 四元数。
旧的 ``@GET_TOOL#`` 三坐标协议仅保留为历史调试兼容。
"""

import json
from typing import Optional, Sequence, Tuple

from coordinate.pose_transform import transform_pose_xyzw

GET_TOOL_COMMAND = "@GET_TOOL#"
GET_TAG_POSE_COMMAND = "@GET_TAG_POSE#"
PROTOCOL_NAME = "sukinee_tag_pose_v1"


def is_get_tool_command(message: str) -> bool:
    """判断是否为旧版工具坐标查询命令。"""
    return message.strip() == GET_TOOL_COMMAND


def is_get_tag_pose_command(message: str) -> bool:
    """判断是否为新版 6D 位姿采样命令。"""
    return message.strip() == GET_TAG_POSE_COMMAND


def format_tag_pose(position_m: Tuple[float, float, float], age_ms: int) -> str:
    """按旧协议格式化 tag1 相对 tag0 平移，输出单位毫米。"""
    x_m, y_m, z_m = position_m
    return "@TOOL,{:.1f},{:.1f},{:.1f},{}#".format(
        x_m * 1000.0,
        y_m * 1000.0,
        z_m * 1000.0,
        int(age_ms),
    )


def format_no_tag() -> str:
    """返回旧协议的无有效 tag 响应。"""
    return "@NO_TAG#"


def format_error(message: str) -> str:
    """返回 ASCII 安全的错误帧，避免破坏串口帧头帧尾。"""
    safe_message = str(message).replace(",", " ").replace("#", " ").replace("@", " ")
    return "@ERR,{}#".format(safe_message.strip())


def format_pose_sample(
    sample_id: str,
    seq: int,
    timestamp_jetson: str,
    transform: Sequence[Sequence[float]],
    tag_size_m: float,
    frame_count_used: int,
    base_ref_seen: bool,
    tool0_seen: bool,
    base_ref_source: str = "live",
    decision_margin_min: Optional[float] = None,
    hamming_max: Optional[int] = None,
    mean_reprojection_error_px: Optional[float] = None,
    tag_family: str = "tag25h9",
    tag_base_ref_id: int = 0,
    tag_tool0_id: int = 1,
    crc32: Optional[str] = None,
) -> str:
    """将有效 6D 位姿样本格式化为单行 JSON。"""
    position_m, orientation_xyzw = transform_pose_xyzw(transform)
    payload = _base_payload(
        sample_id,
        seq,
        timestamp_jetson,
        tag_size_m,
        frame_count_used,
        tag_family,
        tag_base_ref_id,
        tag_tool0_id,
        crc32,
    )
    payload["position_m"] = _round_list(position_m)
    payload["orientation_xyzw"] = _round_list(orientation_xyzw)
    payload["quality"] = _quality_payload(
        base_ref_seen=base_ref_seen,
        tool0_seen=tool0_seen,
        base_ref_source=base_ref_source,
        mean_reprojection_error_px=mean_reprojection_error_px,
        decision_margin_min=decision_margin_min,
        hamming_max=hamming_max,
    )
    return _json_line(payload)


def format_invalid_pose_sample(
    sample_id: str,
    seq: int,
    timestamp_jetson: str,
    tag_size_m: float,
    frame_count_used: int,
    base_ref_seen: bool,
    tool0_seen: bool,
    base_ref_source: str = "none",
    tag_family: str = "tag25h9",
    tag_base_ref_id: int = 0,
    tag_tool0_id: int = 1,
    crc32: Optional[str] = None,
) -> str:
    """将无效采样格式化为 JSON，并明确质量状态。"""
    payload = _base_payload(
        sample_id,
        seq,
        timestamp_jetson,
        tag_size_m,
        frame_count_used,
        tag_family,
        tag_base_ref_id,
        tag_tool0_id,
        crc32,
    )
    payload["position_m"] = None
    payload["orientation_xyzw"] = None
    payload["quality"] = _quality_payload(
        base_ref_seen=base_ref_seen,
        tool0_seen=tool0_seen,
        base_ref_source=base_ref_source,
        mean_reprojection_error_px=None,
        decision_margin_min=None,
        hamming_max=None,
    )
    return _json_line(payload)


def _base_payload(
    sample_id,
    seq,
    timestamp_jetson,
    tag_size_m,
    frame_count_used,
    tag_family,
    tag_base_ref_id,
    tag_tool0_id,
    crc32,
):
    """构造协议公共字段。"""
    return {
        "protocol": PROTOCOL_NAME,
        "sample_id": str(sample_id),
        "seq": int(seq),
        "timestamp_jetson": str(timestamp_jetson),
        "from_frame": "tag_base_ref",
        "to_frame": "tag_tool0",
        "tag_family": str(tag_family),
        "tag_base_ref_id": int(tag_base_ref_id),
        "tag_tool0_id": int(tag_tool0_id),
        "tag_size_m": float(tag_size_m),
        "frame_count_used": int(frame_count_used),
        "crc32": crc32,
    }


def _quality_payload(
    base_ref_seen,
    tool0_seen,
    base_ref_source,
    mean_reprojection_error_px,
    decision_margin_min,
    hamming_max,
):
    """构造质量字段，供电脑端判断本次样本是否可用于标定。"""
    return {
        "both_tags_seen": bool(tool0_seen and base_ref_source in ("live", "cached", "mixed")),
        "base_ref_seen": bool(base_ref_seen),
        "tool0_seen": bool(tool0_seen),
        "base_ref_source": str(base_ref_source),
        "mean_reprojection_error_px": mean_reprojection_error_px,
        "decision_margin_min": decision_margin_min,
        "hamming_max": hamming_max,
    }


def _round_list(values):
    """限制浮点输出长度，减少串口日志噪音。"""
    return [round(float(value), 10) for value in values]


def _json_line(payload):
    """生成单行 JSON，便于电脑端按换行读取。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
