"""保存 RGB-Depth 对齐验证数据。"""

import json
from pathlib import Path


def create_capture_directory(root, timestamp):
    """创建一次验证数据的时间戳目录。"""
    capture_dir = Path(root) / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)
    return capture_dir


def save_measurement(output_path, pixel, depth_m, camera_point):
    """保存点击像素、深度和相机三维坐标。"""
    record = {
        "status": "有效" if camera_point is not None else "无有效深度",
        "pixel": {"u": pixel[0], "v": pixel[1]},
        "depth_m": depth_m,
        "camera_point_m": None,
    }
    if camera_point is not None:
        record["camera_point_m"] = {
            "x": camera_point[0],
            "y": camera_point[1],
            "z": camera_point[2],
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
