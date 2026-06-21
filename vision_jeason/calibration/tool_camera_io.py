"""手眼标定外参 ``T_tool_camera`` 的读写工具。

控制侧尚未能实时发送 ``T_base_tool`` 时，视觉侧仍然可以先用本模块读取
手眼外参占位文件，并配合模拟末端位姿验证完整 eye-in-hand 坐标链路。
文件扩展名使用 ``.yaml``，内容采用项目可控的简化键值格式，避免在
Jetson Nano Python 3.6 环境中额外引入 PyYAML 依赖。
"""

import json
from pathlib import Path

from coordinate.frame_transform import make_transform_from_pose_xyzw

SCHEMA = "zhufeng_tool_camera_v1"


def save_tool_camera_record(record, output_path):
    """保存手眼外参记录。

    ``translation_m`` 表示相机坐标系原点在工具坐标系下的位置，单位为米；
    ``orientation_xyzw`` 使用 ``x, y, z, w`` 顺序保存四元数。该顺序与工程内
    其它姿态解析模块保持一致，避免和 ROS 常见顺序混淆。
    """

    normalized = _normalize_record(record)
    lines = [
        "schema: {}".format(normalized["schema"]),
        "unit: {}".format(normalized["unit"]),
        "translation_m: {}".format(_format_float_list(normalized["translation_m"])),
        "orientation_xyzw: {}".format(_format_float_list(normalized["orientation_xyzw"])),
        "captured_at: {}".format(normalized.get("captured_at", "")),
        "source: {}".format(normalized.get("source", "")),
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_tool_camera_record(input_path):
    """读取手眼外参文件，并附带生成可直接使用的 4x4 变换矩阵。"""

    input_path = Path(input_path)
    data = _load_record_text(input_path.read_text(encoding="utf-8"))
    record = _normalize_record(data)
    x_m, y_m, z_m = record["translation_m"]
    qx, qy, qz, qw = record["orientation_xyzw"]
    record["transform"] = make_transform_from_pose_xyzw(x_m, y_m, z_m, qx, qy, qz, qw)
    return record


def _normalize_record(record):
    """把外部输入整理成稳定字段，并校验关键字段存在。"""

    translation = _as_float_tuple(record.get("translation_m"), 3, "translation_m")
    orientation = _as_float_tuple(record.get("orientation_xyzw"), 4, "orientation_xyzw")
    return {
        "schema": record.get("schema") or SCHEMA,
        "unit": record.get("unit") or "meter",
        "translation_m": translation,
        "orientation_xyzw": orientation,
        "captured_at": record.get("captured_at", ""),
        "source": record.get("source", ""),
    }


def _load_record_text(text):
    """解析 JSON 或简化 YAML 键值文本。"""

    stripped = text.strip()
    if not stripped:
        raise ValueError("手眼外参文件为空")
    if stripped.startswith("{"):
        return json.loads(stripped)

    record = {}
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("无法解析手眼外参行：{}".format(raw_line))
        key, value = line.split(":", 1)
        record[key.strip()] = _parse_scalar(value.strip())
    return record


def _parse_scalar(value):
    """解析一行键值中的标量或数组。"""

    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value)
    try:
        return float(value)
    except ValueError:
        return value


def _as_float_tuple(values, expected_count, field_name):
    """校验数组长度并转换为浮点元组。"""

    if values is None:
        raise ValueError("缺少字段 {}".format(field_name))
    if len(values) != expected_count:
        raise ValueError("{} 需要 {} 个数值".format(field_name, expected_count))
    return tuple(float(value) for value in values)


def _format_float_list(values):
    """输出 JSON 数组格式，方便人工编辑和脚本解析。"""

    return json.dumps([float(value) for value in values], ensure_ascii=False)
