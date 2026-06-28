"""Jetson 高性能 YOLO + 深度 3D 场景识别入口。

该脚本面向正式部署测试：采集一帧 D435 对齐图像，使用 YOLO/TensorRT 识别物体，
再只在检测框内部生成深度 3D 几何和 GRASP 建议。默认输出 JSON，同时可打印现有
``@OBJ`` / ``@GRASP`` 协议帧，方便后续接入串口服务。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.tool_camera_io import load_tool_camera_record
from camera.realsense_camera import RealSenseCamera
from communication.pose_source import load_base_tool_pose_from_file
from communication.protocol import (
    format_end,
    format_grasp_candidate,
    format_no_object,
    format_object_candidate,
)
from coordinate.frame_transform import compose_transform
from perception.grasp_planner import build_visibility_aware_grasps
from perception.yolo_depth_geometry import build_yolo_depth_candidates
from tools.benchmark_yolo_depth_jetson import _detections_from_ultralytics_result
from tools.capture_eye_in_hand_debug import intrinsics_to_dict


DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n_manual_best.engine"
DEFAULT_FALLBACK_MODEL = PROJECT_ROOT / "models" / "yolov8n_manual_best.pt"
DEFAULT_OUTPUT_ROOT = Path("/mnt/zhufeng_data/data/yolo_depth_scene")
DEFAULT_TOOL_CAMERA = PROJECT_ROOT / "calibration" / "tool_camera.example.yaml"
DEFAULT_BASE_TOOL_POSE = PROJECT_ROOT / "tools" / "base_tool_pose.example.txt"
DEFAULT_GRIPPER_MAX_OPENING_M = 0.08


def build_yolo_depth_scene_result(
    detections,
    depth_m,
    intrinsics,
    base_from_camera,
    image_size,
    min_depth_points=20,
    depth_stride=2,
    min_visibility=0.60,
    gripper_max_opening_m=DEFAULT_GRIPPER_MAX_OPENING_M,
):
    """从 YOLO 检测和深度图构建完整场景结果。

    返回结构与现有 eye-in-hand 调试结果保持接近：包含候选物、抓取建议、数量统计和
    坐标系说明。后续串口服务可以直接消费 ``candidates`` 和 ``grasps``。
    """

    candidates = build_yolo_depth_candidates(
        detections,
        depth_m=depth_m,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        min_points=min_depth_points,
        stride=depth_stride,
    )
    grasps = build_visibility_aware_grasps(
        candidates,
        image_size=image_size,
        min_visibility=min_visibility,
        gripper_max_opening_m=gripper_max_opening_m,
    )
    return {
        "source": "yolo_depth",
        "frame": "base",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "grasp_count": len(grasps),
        "rejected_grasp_count": len(candidates) - len(grasps),
        "grasps": grasps,
    }


def format_scene_protocol_frames(result):
    """把场景结果转换成现有 ASCII 串口协议帧。"""

    frames = []
    for candidate in result.get("candidates", []):
        frames.append(format_object_candidate(candidate))
    for grasp in result.get("grasps", []):
        frames.append(format_grasp_candidate(grasp))
    if not frames:
        frames.append(format_no_object())
    frames.append(format_end(len(frames) - 1 if frames[0] == format_no_object() else len(frames)))
    return frames


def run_single_scene(
    model_path=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    conf=0.50,
    imgsz=640,
    width=640,
    height=480,
    fps=30,
    min_depth_points=20,
    depth_stride=2,
    print_protocol=False,
    base_from_tool=None,
):
    """在 Jetson 上采集并识别一帧，返回保存后的场景结果。"""

    import numpy as np
    from ultralytics import YOLO

    selected_model = _select_model_path(model_path)
    model = YOLO(str(selected_model))
    base_from_camera = _load_base_from_camera(
        DEFAULT_TOOL_CAMERA,
        DEFAULT_BASE_TOOL_POSE,
        base_from_tool=base_from_tool,
    )
    camera = RealSenseCamera(width=width, height=height, fps=fps)
    camera.start()
    try:
        intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
        depth_scale = float(camera.get_depth_scale())
        started_at = perf_counter()
        color_frame, depth_frame = camera.capture_aligned()
        captured_at = perf_counter()
        color_bgr = np.asanyarray(color_frame.get_data())
        yolo_result = model.predict(color_bgr, imgsz=int(imgsz), conf=float(conf), verbose=False)[0]
        inferred_at = perf_counter()
        detections = _detections_from_ultralytics_result(yolo_result)
        depth_m = np.asanyarray(depth_frame.get_data()).astype("float32") * depth_scale
        result = build_yolo_depth_scene_result(
            detections=detections,
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            image_size=(int(width), int(height)),
            min_depth_points=min_depth_points,
            depth_stride=depth_stride,
        )
        finished_at = perf_counter()
        result.update(
            {
                "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "model_path": str(selected_model),
                "conf": float(conf),
                "imgsz": int(imgsz),
                "timing_ms": {
                    "capture": _elapsed_ms(started_at, captured_at),
                    "yolo": _elapsed_ms(captured_at, inferred_at),
                    "geometry": _elapsed_ms(inferred_at, finished_at),
                    "total": _elapsed_ms(started_at, finished_at),
                },
            }
        )
        output_path = _save_scene_result(output_root, result)
        result["output_path"] = str(output_path)
        _save_scene_result(output_root, result, fixed_path=output_path)
        if print_protocol:
            for frame in format_scene_protocol_frames(result):
                print(frame)
        else:
            print("YOLO 深度场景结果已保存：{0}".format(output_path))
            print("OBJ={0} GRASP={1} total_ms={2:.1f}".format(
                result["candidate_count"],
                result["grasp_count"],
                result["timing_ms"]["total"],
            ))
        return result
    finally:
        camera.stop()


def _select_model_path(model_path):
    """优先使用 TensorRT engine；如果还没导出，则回退到项目内 .pt 权重。"""

    if model_path is not None:
        selected = Path(model_path)
    elif DEFAULT_MODEL.exists():
        selected = DEFAULT_MODEL
    else:
        selected = DEFAULT_FALLBACK_MODEL
    if not selected.exists():
        raise FileNotFoundError("找不到 YOLO 模型文件：{0}".format(selected))
    return selected


def _load_base_from_camera(tool_camera_path, base_tool_pose_path, base_from_tool=None):
    """加载手眼外参和末端位姿，组合成 ``T_base_camera``。"""

    tool_camera = load_tool_camera_record(tool_camera_path)
    if base_from_tool is None:
        base_tool_pose = load_base_tool_pose_from_file(base_tool_pose_path)
        base_from_tool = base_tool_pose["transform"]
    return compose_transform(base_from_tool, tool_camera["transform"])


def _save_scene_result(output_root, result, fixed_path=None):
    """保存单帧识别结果 JSON。"""

    if fixed_path is None:
        output_dir = Path(output_root) / result["created_at"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "yolo_depth_scene.json"
    else:
        output_path = Path(fixed_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _elapsed_ms(start, end):
    """把 ``perf_counter`` 差值转换为毫秒。"""

    return (float(end) - float(start)) * 1000.0


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="YOLO .engine 或 .pt 路径")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--depth-stride", type=int, default=2, help="ROI 深度采样步长")
    parser.add_argument("--min-depth-points", type=int, default=20, help="ROI 最少有效深度点")
    parser.add_argument("--protocol", action="store_true", help="打印 @OBJ/@GRASP 协议帧")
    args = parser.parse_args(argv)

    run_single_scene(
        model_path=args.model,
        output_root=args.output_root,
        conf=args.conf,
        imgsz=args.imgsz,
        min_depth_points=args.min_depth_points,
        depth_stride=args.depth_stride,
        print_protocol=args.protocol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
