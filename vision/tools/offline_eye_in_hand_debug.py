"""离线 eye-in-hand 几何链路调试工具。

该脚本用于控制侧暂时不能发送 ``T_base_tool`` 时的 PC/Jetson 离线验证：
读取手眼外参、模拟末端位姿、深度矩阵和相机内参，计算当前
``T_base_camera``，再输出基坐标系下的未知物体候选。
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from calibration.tool_camera_io import load_tool_camera_record
from communication.pose_source import load_base_tool_pose_from_file
from coordinate.frame_transform import compose_transform
from perception.object_fusion import build_base_height_object_candidates


def build_offline_eye_in_hand_result(
    depth_m,
    intrinsics,
    base_from_tool,
    tool_from_camera,
    min_points=20,
    pixel_radius=3,
    stride=1,
    min_z_base_m=0.01,
    max_z_base_m=0.30,
):
    """用模拟位姿跑通一次 eye-in-hand 候选物生成流程。"""

    base_from_camera = compose_transform(base_from_tool, tool_from_camera)
    candidates = build_base_height_object_candidates(
        depth_m=depth_m,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        min_points=min_points,
        pixel_radius=pixel_radius,
        stride=stride,
        min_z_base_m=min_z_base_m,
        max_z_base_m=max_z_base_m,
    )
    return {
        "frame": "base",
        "base_from_camera": base_from_camera,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def load_depth_matrix(input_path):
    """读取离线深度矩阵，支持直接二维数组或 ``{"depth_m": [...]}``。"""

    data = json.loads(_read_json_text(input_path))
    if isinstance(data, dict):
        return data["depth_m"]
    return data


def load_intrinsics(input_path):
    """读取相机内参，支持简化内参文件或 D435 完整内参记录。"""

    data = json.loads(_read_json_text(input_path))
    if "aligned_depth" in data:
        aligned_depth = data["aligned_depth"]
        return {
            "fx": aligned_depth["fx"],
            "fy": aligned_depth["fy"],
            "cx": aligned_depth["cx"],
            "cy": aligned_depth["cy"],
        }
    return {
        "fx": data["fx"],
        "fy": data["fy"],
        "cx": data["cx"],
        "cy": data["cy"],
    }


def _read_json_text(input_path):
    """读取 JSON 文本，兼容 Windows PowerShell 生成的 UTF-8 BOM 文件。"""

    return Path(input_path).read_text(encoding="utf-8-sig")


def save_offline_result(result, output_path):
    """保存离线调试结果，供现场复盘和参数调试使用。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-json", required=True, help="离线深度矩阵 JSON")
    parser.add_argument("--intrinsics-json", required=True, help="相机内参 JSON")
    parser.add_argument("--tool-camera", required=True, help="tool_camera.yaml 路径")
    parser.add_argument("--base-tool-pose", required=True, help="模拟 T_base_tool 位姿文件")
    parser.add_argument("--output-json", required=True, help="调试结果输出 JSON")
    parser.add_argument("--min-points", type=int, default=20, help="候选物聚类最少点数")
    parser.add_argument("--pixel-radius", type=int, default=3, help="像素连通半径")
    parser.add_argument("--stride", type=int, default=1, help="深度图采样步长")
    parser.add_argument("--min-z-base", type=float, default=0.01, help="基坐标系最小高度，单位米")
    parser.add_argument("--max-z-base", type=float, default=0.30, help="基坐标系最大高度，单位米")
    args = parser.parse_args(argv)

    tool_camera = load_tool_camera_record(args.tool_camera)
    base_tool_pose = load_base_tool_pose_from_file(args.base_tool_pose)
    result = build_offline_eye_in_hand_result(
        depth_m=load_depth_matrix(args.depth_json),
        intrinsics=load_intrinsics(args.intrinsics_json),
        base_from_tool=base_tool_pose["transform"],
        tool_from_camera=tool_camera["transform"],
        min_points=args.min_points,
        pixel_radius=args.pixel_radius,
        stride=args.stride,
        min_z_base_m=args.min_z_base,
        max_z_base_m=args.max_z_base,
    )
    save_offline_result(result, args.output_json)
    print("离线 eye-in-hand 调试结果已保存到：{}".format(args.output_json))


if __name__ == "__main__":
    main()
