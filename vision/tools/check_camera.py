"""在 Jetson 上检查 RealSense 彩色帧、对齐深度帧和中心点深度。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.realsense_camera import RealSenseCamera


def run(frame_limit=None):
    """持续打印对齐帧信息；frame_limit 用于有限次冒烟测试。"""
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
