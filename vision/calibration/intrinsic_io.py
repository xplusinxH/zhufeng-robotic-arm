"""构造并保存 D435 内参记录。"""

import json
from pathlib import Path


def _intrinsics_to_dict(intrinsics):
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
    """构造可序列化的 D435 内参记录。"""
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
    """将内参记录保存为 UTF-8 JSON。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
