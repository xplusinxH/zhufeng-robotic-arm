"""采集一帧 D435 深度并运行 eye-in-hand 候选物调试。

控制侧暂时不能实时发送 ``T_base_tool`` 时，本工具使用手动位姿文件模拟
末端位姿，配合 ``tool_camera.yaml`` 手眼外参，在真实 D435 单帧深度上
验证 ``P_camera -> P_base`` 与桌面上方候选物提取链路。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from calibration.tool_camera_io import load_tool_camera_record
from camera.realsense_camera import RealSenseCamera
from communication.pose_source import load_base_tool_pose_from_file
from tools.offline_eye_in_hand_debug import build_offline_eye_in_hand_result

DEFAULT_OUTPUT_ROOT = Path("/mnt/zhufeng_data/zhufeng/data/eye_in_hand_debug")


def capture_eye_in_hand_debug(
    camera,
    base_from_tool,
    tool_from_camera,
    output_root,
    timestamp=None,
    min_points=20,
    pixel_radius=3,
    stride=1,
    min_z_base_m=0.01,
    max_z_base_m=0.30,
):
    """采集一帧对齐深度，并保存基坐标系候选物调试结果。

    ``camera`` 只需要满足 ``RealSenseCamera`` 的最小接口，因此 PC 单元测试
    可以注入假相机；Jetson 真机运行时则传入真实 ``RealSenseCamera``。
    """

    camera.start()
    try:
        _color_frame, depth_frame = camera.capture_aligned()
        intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
        depth_m = depth_frame_to_depth_m(depth_frame)
        result = build_offline_eye_in_hand_result(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_tool=base_from_tool,
            tool_from_camera=tool_from_camera,
            min_points=min_points,
            pixel_radius=pixel_radius,
            stride=stride,
            min_z_base_m=min_z_base_m,
            max_z_base_m=max_z_base_m,
        )
        result["intrinsics"] = intrinsics
        result["depth_size"] = {
            "width": len(depth_m[0]) if depth_m else 0,
            "height": len(depth_m),
        }
        result["captured_at"] = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = save_capture_result(output_root, result["captured_at"], result)
        result["output_path"] = str(output_path)
        _rewrite_result_with_output_path(output_path, result)
        return result
    finally:
        camera.stop()


def intrinsics_to_dict(intrinsics):
    """把 RealSense 内参对象转换为几何函数使用的字典。"""

    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
    }


def depth_frame_to_depth_m(depth_frame):
    """把 RealSense 深度帧转换为米单位二维数组。

    单帧调试优先追求稳和容易排查，直接使用 SDK 的 ``get_distance`` 读取每个
    像素。后续若需要连续实时运行，再替换为 raw buffer 批量转换路径。
    """

    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    return [
        [float(depth_frame.get_distance(u, v)) for u in range(width)]
        for v in range(height)
    ]


def save_capture_result(output_root, timestamp, result):
    """将本次单帧调试结果保存为 JSON。"""

    capture_dir = Path(output_root) / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)
    output_path = capture_dir / "eye_in_hand_candidates.json"
    _rewrite_result_with_output_path(output_path, result)
    return output_path


def _rewrite_result_with_output_path(output_path, result):
    """写出 UTF-8 JSON，Windows 和 Jetson 均可直接查看。"""

    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-camera", required=True, help="tool_camera.yaml 路径")
    parser.add_argument("--base-tool-pose", required=True, help="模拟 T_base_tool 位姿文件")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=640, help="D435 图像宽度")
    parser.add_argument("--height", type=int, default=480, help="D435 图像高度")
    parser.add_argument("--fps", type=int, default=30, help="D435 帧率")
    parser.add_argument("--min-points", type=int, default=20, help="候选物聚类最少点数")
    parser.add_argument("--pixel-radius", type=int, default=3, help="像素连通半径")
    parser.add_argument("--stride", type=int, default=1, help="深度图采样步长")
    parser.add_argument("--min-z-base", type=float, default=0.01, help="基坐标系最小高度，单位米")
    parser.add_argument("--max-z-base", type=float, default=0.30, help="基坐标系最大高度，单位米")
    args = parser.parse_args(argv)

    tool_camera = load_tool_camera_record(args.tool_camera)
    base_tool_pose = load_base_tool_pose_from_file(args.base_tool_pose)
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)
    result = capture_eye_in_hand_debug(
        camera=camera,
        base_from_tool=base_tool_pose["transform"],
        tool_from_camera=tool_camera["transform"],
        output_root=args.output_root,
        min_points=args.min_points,
        pixel_radius=args.pixel_radius,
        stride=args.stride,
        min_z_base_m=args.min_z_base,
        max_z_base_m=args.max_z_base,
    )
    print("单帧 eye-in-hand 调试结果已保存到：{}".format(result["output_path"]))
    print("候选物数量：{}".format(result["candidate_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
