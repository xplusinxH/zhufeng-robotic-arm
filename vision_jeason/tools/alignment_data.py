"""保存 RGB-Depth 对齐验证数据。

该模块只负责文件目录和 JSON 记录，不依赖 OpenCV 或 RealSense。
实时工具采集到图像后调用这里保存测量结果，便于 PC 端单元测试。
"""

import json
from pathlib import Path


def create_capture_directory(root, timestamp):
    """创建一次验证数据的时间戳目录。"""
    capture_dir = Path(root) / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)
    return capture_dir


def save_measurement(output_path, pixel, depth_m, camera_point):
    """保存点击像素、深度和相机三维坐标。

    ``camera_point`` 为 ``None`` 时表示该像素没有有效深度，JSON 中会写入
    明确状态，避免后续复盘时误判为程序没有保存成功。
    """
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
