"""Jetson 上实测 YOLO TensorRT 识别和深度 3D 几何耗时。

该脚本用于回答一个核心问题：在真实 Jetson + D435 上，视觉端从采集画面到输出
候选物三维坐标到底需要多少毫秒。它不会打开可视化窗口，避免 GUI 干扰性能统计。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from calibration.tool_camera_io import load_tool_camera_record
from camera.realsense_camera import RealSenseCamera
from communication.pose_source import load_base_tool_pose_from_file
from coordinate.frame_transform import compose_transform
from perception.yolo_depth_geometry import build_yolo_depth_candidates
from tools.capture_eye_in_hand_debug import intrinsics_to_dict


DEFAULT_ENGINE = PROJECT_ROOT / "models" / "yolov8n_manual_best.engine"
DEFAULT_OUTPUT_ROOT = Path("/mnt/zhufeng_data/data/yolo_jetson_benchmark")
DEFAULT_TOOL_CAMERA = PROJECT_ROOT / "calibration" / "tool_camera.example.yaml"
DEFAULT_BASE_TOOL_POSE = PROJECT_ROOT / "tools" / "base_tool_pose.example.txt"


def benchmark_yolo_depth(
    model_path,
    output_root=DEFAULT_OUTPUT_ROOT,
    frames=100,
    warmup=10,
    conf=0.50,
    imgsz=640,
    width=640,
    height=480,
    fps=30,
    min_depth_points=20,
    depth_stride=2,
    tool_camera_path=DEFAULT_TOOL_CAMERA,
    base_tool_pose_path=DEFAULT_BASE_TOOL_POSE,
):
    """运行真实 D435 + YOLO 模型测速，并返回统计结果。"""

    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    camera = RealSenseCamera(width=width, height=height, fps=fps)
    camera.start()
    try:
        intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
        depth_scale = float(camera.get_depth_scale())
        base_from_camera = _load_base_from_camera(tool_camera_path, base_tool_pose_path)
        records = []
        last_candidates = []
        total_frames = int(warmup) + int(frames)
        for index in range(total_frames):
            frame_start = perf_counter()
            color_frame, depth_frame = camera.capture_aligned()
            capture_done = perf_counter()

            color_bgr = np.asanyarray(color_frame.get_data())
            result = model.predict(color_bgr, imgsz=int(imgsz), conf=float(conf), verbose=False)[0]
            inference_done = perf_counter()

            detections = _detections_from_ultralytics_result(result)
            depth_m = np.asanyarray(depth_frame.get_data()).astype("float32") * depth_scale
            candidates = build_yolo_depth_candidates(
                detections,
                depth_m=depth_m,
                intrinsics=intrinsics,
                base_from_camera=base_from_camera,
                min_points=min_depth_points,
                stride=depth_stride,
            )
            geometry_done = perf_counter()

            if index >= int(warmup):
                records.append(
                    {
                        "capture_ms": _elapsed_ms(frame_start, capture_done),
                        "yolo_ms": _elapsed_ms(capture_done, inference_done),
                        "geometry_ms": _elapsed_ms(inference_done, geometry_done),
                        "total_ms": _elapsed_ms(frame_start, geometry_done),
                        "detection_count": len(detections),
                        "candidate_count": len(candidates),
                    }
                )
                last_candidates = candidates
                _print_frame_record(len(records), records[-1])
        summary = _build_summary(records)
        payload = {
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model_path": str(model_path),
            "frame_count": len(records),
            "warmup": int(warmup),
            "conf": float(conf),
            "imgsz": int(imgsz),
            "depth_stride": int(depth_stride),
            "min_depth_points": int(min_depth_points),
            "summary": summary,
            "last_candidates": last_candidates,
            "records": records,
        }
        output_path = _save_benchmark_result(output_root, payload)
        payload["output_path"] = str(output_path)
        return payload
    finally:
        camera.stop()


def _load_base_from_camera(tool_camera_path, base_tool_pose_path):
    """从手眼外参和当前末端位姿文件计算 ``T_base_camera``。"""

    tool_camera_path = Path(tool_camera_path)
    base_tool_pose_path = Path(base_tool_pose_path)
    if not tool_camera_path.exists() or not base_tool_pose_path.exists():
        return None
    tool_camera = load_tool_camera_record(tool_camera_path)
    base_tool_pose = load_base_tool_pose_from_file(base_tool_pose_path)
    return compose_transform(base_tool_pose["transform"], tool_camera["transform"])


def _detections_from_ultralytics_result(result):
    """把 Ultralytics 推理结果转成项目内部检测字典。"""

    detections = []
    names = result.names or {}
    if result.boxes is None:
        return detections
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for bbox, confidence, class_id in zip(boxes_xyxy, confidences, classes):
        detections.append(
            {
                "bbox_pixel": tuple(float(value) for value in bbox.tolist()),
                "score": float(confidence),
                "class_id": int(class_id),
                "class_name": str(names.get(int(class_id), int(class_id))),
            }
        )
    return detections


def _build_summary(records):
    """生成平均值、P50、P95，方便现场快速判断瓶颈。"""

    if not records:
        return {}
    fields = ("capture_ms", "yolo_ms", "geometry_ms", "total_ms")
    summary = {}
    for field in fields:
        values = [float(item[field]) for item in records]
        summary[field] = {
            "avg": float(statistics.fmean(values)),
            "p50": _percentile(values, 50.0),
            "p95": _percentile(values, 95.0),
            "max": float(max(values)),
        }
    summary["avg_detection_count"] = float(
        statistics.fmean(float(item["detection_count"]) for item in records)
    )
    summary["avg_candidate_count"] = float(
        statistics.fmean(float(item["candidate_count"]) for item in records)
    )
    return summary


def _percentile(values, percentile):
    """不依赖新版 NumPy 的轻量百分位数计算。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * float(percentile) / 100.0
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _save_benchmark_result(output_root, payload):
    """保存测速结果 JSON。"""

    output_dir = Path(output_root) / payload["created_at"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "yolo_depth_benchmark.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _print_frame_record(index, record):
    """逐帧打印简短耗时，便于现场观察是否有偶发卡顿。"""

    print(
        "frame={0} total={1:.1f}ms yolo={2:.1f}ms geom={3:.1f}ms det={4} cand={5}".format(
            index,
            record["total_ms"],
            record["yolo_ms"],
            record["geometry_ms"],
            record["detection_count"],
            record["candidate_count"],
        )
    )


def _elapsed_ms(start, end):
    """把 ``perf_counter`` 差值转成毫秒。"""

    return (float(end) - float(start)) * 1000.0


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_ENGINE, help="YOLO .engine 或 .pt 路径")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frames", type=int, default=100, help="正式统计帧数")
    parser.add_argument("--warmup", type=int, default=10, help="预热帧数")
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--depth-stride", type=int, default=2, help="ROI 深度采样步长")
    parser.add_argument("--min-depth-points", type=int, default=20, help="ROI 内最少有效深度点")
    parser.add_argument("--width", type=int, default=640, help="D435 宽度")
    parser.add_argument("--height", type=int, default=480, help="D435 高度")
    parser.add_argument("--fps", type=int, default=30, help="D435 帧率")
    args = parser.parse_args(argv)

    result = benchmark_yolo_depth(
        model_path=args.model,
        output_root=args.output_root,
        frames=args.frames,
        warmup=args.warmup,
        conf=args.conf,
        imgsz=args.imgsz,
        width=args.width,
        height=args.height,
        fps=args.fps,
        min_depth_points=args.min_depth_points,
        depth_stride=args.depth_stride,
    )
    print("测速结果已保存：{0}".format(result["output_path"]))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
