"""在 Jetson 上读取并保存 D435 当前流配置的内参。

输出文件默认写入 ``/etc/zhufeng-vision/calibration/camera_intrinsic.json``。
该脚本必须在 D435 已连接、RealSense SDK 可用的 Jetson 真机上运行。
"""

import argparse
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.intrinsic_io import build_intrinsic_record, save_intrinsic_record
from camera.realsense_camera import RealSenseCamera


DEFAULT_OUTPUT = Path("/etc/zhufeng-vision/calibration/camera_intrinsic.json")


def main():
    """命令行入口：启动相机、读取内参、保存 JSON 后释放相机。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    camera = RealSenseCamera()
    camera.start()
    try:
        device_info = camera.get_device_info()
        record = build_intrinsic_record(
            camera.get_color_intrinsics(),
            camera.get_aligned_depth_intrinsics(),
            camera.get_depth_scale(),
            device_info["serial_number"],
            device_info["firmware_version"],
            datetime.now().astimezone().isoformat(),
        )
        save_intrinsic_record(record, args.output)
    finally:
        camera.stop()

    print("内参已保存到：{}".format(args.output))
    print("彩色内参：{}".format(record["color"]))
    print("对齐深度内参：{}".format(record["aligned_depth"]))
    print("深度比例尺：{} 米".format(record["depth_scale_m"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
