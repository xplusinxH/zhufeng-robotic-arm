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
from perception.grasp_planner import build_visibility_aware_grasps
from perception.object_fusion import (
    build_base_height_object_candidates,
    build_depth_foreground_object_candidates,
    build_table_plane_object_candidates,
)
from perception.table_plane import (
    apply_table_z_compensation_to_scene,
    estimate_table_plane_diagnostics,
)


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
    image_size=None,
    camera_keepout_roi=None,
    min_visibility=0.60,
    segmentation_mode="table_plane",
    enable_table_z_compensation=False,
    table_plane_stride=None,
    table_plane_min_points=80,
    gripper_max_opening_m=None,
):
    """用模拟位姿跑通一次 eye-in-hand 候选物生成流程。"""

    base_from_camera = compose_transform(base_from_tool, tool_from_camera)
    table_plane = estimate_table_plane_diagnostics(
        depth_m=depth_m,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        stride=table_plane_stride or max(1, int(stride)),
        min_points=table_plane_min_points,
    )
    if segmentation_mode == "table_plane":
        candidates = build_table_plane_object_candidates(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=min_points,
            pixel_radius=pixel_radius,
            stride=stride,
            table_plane_min_points=table_plane_min_points,
        )
    elif segmentation_mode == "depth_foreground":
        candidates = build_depth_foreground_object_candidates(
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            min_points=min_points,
            pixel_radius=pixel_radius,
            stride=stride,
        )
    else:
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
    grasps = build_visibility_aware_grasps(
        candidates,
        image_size=image_size or _infer_image_size(depth_m),
        camera_keepout_roi=camera_keepout_roi,
        min_visibility=min_visibility,
        gripper_max_opening_m=gripper_max_opening_m,
    )
    if enable_table_z_compensation:
        apply_table_z_compensation_to_scene(candidates, grasps, table_plane)
    return {
        "frame": "base",
        "base_from_camera": base_from_camera,
        "table_plane": table_plane,
        "table_z_compensation_enabled": bool(enable_table_z_compensation),
        "gripper_max_opening_m": gripper_max_opening_m,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "grasp_count": len(grasps),
        "rejected_grasp_count": len(candidates) - len(grasps),
        "grasps": grasps,
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


def parse_roi(value):
    """解析 ``u1,v1,u2,v2`` 格式的 ROI 参数。"""

    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 必须使用 u1,v1,u2,v2 格式")
    return tuple(int(part) for part in parts)


def _infer_image_size(depth_m):
    """从深度矩阵推断图像尺寸。"""

    if not depth_m:
        return (0, 0)
    return (len(depth_m[0]), len(depth_m))


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
    parser.add_argument("--camera-keepout-roi", help="相机视野禁区 ROI，格式 u1,v1,u2,v2")
    parser.add_argument("--min-visibility", type=float, default=0.60, help="生成 GRASP 的最小视野评分")
    parser.add_argument(
        "--table-z-compensation",
        action="store_true",
        help="使用拟合桌面平面对候选物和 GRASP 的 Z_base 做临时补偿",
    )
    args = parser.parse_args(argv)

    depth_m = load_depth_matrix(args.depth_json)
    tool_camera = load_tool_camera_record(args.tool_camera)
    base_tool_pose = load_base_tool_pose_from_file(args.base_tool_pose)
    result = build_offline_eye_in_hand_result(
        depth_m=depth_m,
        intrinsics=load_intrinsics(args.intrinsics_json),
        base_from_tool=base_tool_pose["transform"],
        tool_from_camera=tool_camera["transform"],
        min_points=args.min_points,
        pixel_radius=args.pixel_radius,
        stride=args.stride,
        min_z_base_m=args.min_z_base,
        max_z_base_m=args.max_z_base,
        image_size=_infer_image_size(depth_m),
        camera_keepout_roi=parse_roi(args.camera_keepout_roi),
        min_visibility=args.min_visibility,
        enable_table_z_compensation=args.table_z_compensation,
    )
    save_offline_result(result, args.output_json)
    print("离线 eye-in-hand 调试结果已保存到：{}".format(args.output_json))


if __name__ == "__main__":
    main()
