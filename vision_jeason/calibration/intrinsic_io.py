"""D435 内参记录的构造与保存。

内参文件是后续像素转三维坐标、桌面平面标定、外参标定的基础数据。
本模块只处理可序列化的数据结构，不直接访问相机硬件。
"""

import json
from pathlib import Path


def _intrinsics_to_dict(intrinsics):
    """将 RealSense SDK 内参对象转换为普通字典。

    字段约定：
    - ``fx/fy/cx/cy`` 均使用像素单位。
    - 畸变参数原样记录，当前几何链路暂不主动做去畸变。
    """
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "distortion_model": str(intrinsics.model),
        "distortion_coefficients": list(intrinsics.coeffs),
    }


def build_intrinsic_record(
    color_intrinsics,
    depth_intrinsics,
    depth_scale,
    serial_number,
    firmware_version,
    captured_at,
):
    """构造可序列化的 D435 内参记录。

    ``depth_scale`` 的单位是米/原始深度单位，例如常见值 ``0.001`` 表示
    深度图整数值 1000 对应 1 米。
    """
    return {
        "captured_at": captured_at,
        "unit": "meter",
        "device": {
            "serial_number": serial_number,
            "firmware_version": firmware_version,
        },
        "color": _intrinsics_to_dict(color_intrinsics),
        "aligned_depth": _intrinsics_to_dict(depth_intrinsics),
        "depth_scale_m": depth_scale,
    }


def save_intrinsic_record(record, output_path):
    """将内参记录保存为 UTF-8 JSON。

    保存前自动创建父目录，便于 Jetson 端直接写入
    ``/etc/zhufeng-vision/calibration``。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
