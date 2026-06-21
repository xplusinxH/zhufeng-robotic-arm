"""在 Jetson 上检查 RealSense 彩色帧、对齐深度帧和中心点深度。

这是最小相机冒烟测试：不显示窗口，只持续打印分辨率和中心点深度。
适合 SSH 终端、无桌面环境或十分钟稳定性测试。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.realsense_camera import RealSenseCamera


def run(frame_limit=None):
    """持续打印对齐帧信息。

    ``frame_limit`` 用于自动化测试或短时间冒烟测试；为 ``None`` 时持续运行，
    直到用户按 Ctrl+C。
    """
    camera = RealSenseCamera()
    frame_number = 0
    camera.start()
    try:
        while frame_limit is None or frame_number < frame_limit:
            color_frame, depth_frame = camera.capture_aligned()
            frame_number += 1
            center_depth = depth_frame.get_distance(
                color_frame.get_width() // 2,
                color_frame.get_height() // 2,
            )
            print(
                "帧 {}：彩色 {}x{}，深度 {}x{}，中心深度 {:.3f} 米".format(
                    frame_number,
                    color_frame.get_width(),
                    color_frame.get_height(),
                    depth_frame.get_width(),
                    depth_frame.get_height(),
                    center_depth,
                )
            )
    finally:
        camera.stop()


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, help="采集指定帧数后退出")
    args = parser.parse_args()
    try:
        run(args.frames)
    except KeyboardInterrupt:
        print("\n用户终止采集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
