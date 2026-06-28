"""桌面抓取视觉主程序。

该入口面向明天的 Jetson 真机测试：默认使用当前手工标注训练得到的
YOLOv8n 模型，采集 RealSense D435 彩色/深度对齐帧，输出物体类别、
相机/基座坐标、几何尺寸和抓取建议。实时模式只显示相机画面，按 D
才执行一次检测，避免 Jetson 在预览阶段持续满负载。
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

from camera.realsense_camera import RealSenseCamera
from tools.benchmark_yolo_depth_jetson import _detections_from_ultralytics_result
from tools.capture_eye_in_hand_debug import intrinsics_to_dict
from tools.run_yolo_depth_scene import (
    DEFAULT_BASE_TOOL_POSE,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOOL_CAMERA,
    _elapsed_ms,
    _load_base_from_camera,
    build_yolo_depth_scene_result,
    format_scene_protocol_frames,
)


DEFAULT_CONF = 0.50
DEFAULT_IMGSZ = 640
DEFAULT_DEPTH_STRIDE = 2
DEFAULT_MIN_DEPTH_POINTS = 20
WINDOW_NAME = "desktop vision"


def choose_model_path(requested_model, engine_path=DEFAULT_MODEL, fallback_path=DEFAULT_FALLBACK_MODEL):
    """选择实际使用的 YOLO 模型路径。

    优先级为：用户显式传入路径 > TensorRT engine > PyTorch pt。这样同一条
    命令在 PC 和 Jetson 上都能使用；Jetson 导出 engine 后会自动优先走加速模型。
    """

    if requested_model is not None:
        selected = Path(requested_model)
    elif Path(engine_path).exists():
        selected = Path(engine_path)
    else:
        selected = Path(fallback_path)
    if not selected.exists():
        raise FileNotFoundError("找不到 YOLO 模型文件：{0}".format(selected))
    return selected


def build_status_text(result):
    """生成窗口底部和终端使用的简短检测状态文本。"""

    if not result:
        return "Ready: press D to detect, S to save, Q to quit"
    timing = result.get("timing_ms", {})
    total_ms = float(timing.get("total", 0.0))
    return "OBJ={0} GRASP={1} total={2:.1f}ms".format(
        int(result.get("candidate_count", 0)),
        int(result.get("grasp_count", 0)),
        total_ms,
    )


def run_once(
    model_path=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    conf=DEFAULT_CONF,
    imgsz=DEFAULT_IMGSZ,
    width=640,
    height=480,
    fps=30,
    depth_stride=DEFAULT_DEPTH_STRIDE,
    min_depth_points=DEFAULT_MIN_DEPTH_POINTS,
    protocol=False,
):
    """采集并识别单帧，适合明天先做无窗口冒烟测试。"""

    import numpy as np
    from ultralytics import YOLO

    selected_model = choose_model_path(model_path)
    model = YOLO(str(selected_model))
    camera = RealSenseCamera(width=width, height=height, fps=fps)
    camera.start()
    try:
        base_from_camera = _load_base_from_camera(DEFAULT_TOOL_CAMERA, DEFAULT_BASE_TOOL_POSE)
        intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
        depth_scale = float(camera.get_depth_scale())
        color_frame, depth_frame = camera.capture_aligned()
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype("float32") * depth_scale
        result = detect_frame(
            model=model,
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            model_path=selected_model,
            conf=conf,
            imgsz=imgsz,
            depth_stride=depth_stride,
            min_depth_points=min_depth_points,
        )
        output_path = save_result(output_root, result)
        result["output_path"] = str(output_path)
        save_result(output_root, result, fixed_path=output_path)
        if protocol:
            for frame in format_scene_protocol_frames(result):
                print(frame)
        else:
            print("视觉结果已保存：{0}".format(output_path))
            print(build_status_text(result))
        return result
    finally:
        camera.stop()


def run_live(
    model_path=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    conf=DEFAULT_CONF,
    imgsz=DEFAULT_IMGSZ,
    width=640,
    height=480,
    fps=30,
    depth_stride=DEFAULT_DEPTH_STRIDE,
    min_depth_points=DEFAULT_MIN_DEPTH_POINTS,
):
    """启动实时预览；按 D 检测，按 S 保存当前画面，按 Q 或 Esc 退出。"""

    import cv2
    import numpy as np
    from ultralytics import YOLO

    selected_model = choose_model_path(model_path)
    model = YOLO(str(selected_model))
    camera = RealSenseCamera(width=width, height=height, fps=fps)
    camera.start()
    latest_result = None
    try:
        base_from_camera = _load_base_from_camera(DEFAULT_TOOL_CAMERA, DEFAULT_BASE_TOOL_POSE)
        intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
        depth_scale = float(camera.get_depth_scale())
        while True:
            color_frame, depth_frame = camera.capture_aligned()
            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            display = color_bgr.copy()
            draw_status(display, build_status_text(latest_result))
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                saved_path = save_image(output_root, color_bgr)
                print("当前画面已保存：{0}".format(saved_path))
            if key in (ord("d"), ord("D")):
                depth_m = np.asanyarray(depth_frame.get_data()).astype("float32") * depth_scale
                latest_result = detect_frame(
                    model=model,
                    color_bgr=color_bgr,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    base_from_camera=base_from_camera,
                    model_path=selected_model,
                    conf=conf,
                    imgsz=imgsz,
                    depth_stride=depth_stride,
                    min_depth_points=min_depth_points,
                )
                output_path = save_result(output_root, latest_result)
                latest_result["output_path"] = str(output_path)
                save_result(output_root, latest_result, fixed_path=output_path)
                print(build_status_text(latest_result))
                print("视觉结果已保存：{0}".format(output_path))
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def detect_frame(
    model,
    color_bgr,
    depth_m,
    intrinsics,
    base_from_camera,
    model_path,
    conf=DEFAULT_CONF,
    imgsz=DEFAULT_IMGSZ,
    depth_stride=DEFAULT_DEPTH_STRIDE,
    min_depth_points=DEFAULT_MIN_DEPTH_POINTS,
):
    """对一帧已经对齐的彩色/深度图执行 YOLO + 深度几何。"""

    started_at = perf_counter()
    yolo_result = model.predict(color_bgr, imgsz=int(imgsz), conf=float(conf), verbose=False)[0]
    inferred_at = perf_counter()
    detections = _detections_from_ultralytics_result(yolo_result)
    height, width = color_bgr.shape[:2]
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
            "model_path": str(model_path),
            "conf": float(conf),
            "imgsz": int(imgsz),
            "timing_ms": {
                "yolo": _elapsed_ms(started_at, inferred_at),
                "geometry": _elapsed_ms(inferred_at, finished_at),
                "total": _elapsed_ms(started_at, finished_at),
            },
        }
    )
    return result


def draw_status(image_bgr, text):
    """在预览图左上角绘制轻量状态条。"""

    import cv2

    cv2.rectangle(image_bgr, (0, 0), (int(image_bgr.shape[1]), 28), (0, 0, 0), -1)
    cv2.putText(image_bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def save_result(output_root, result, fixed_path=None):
    """保存单次检测 JSON。"""

    if fixed_path is None:
        output_dir = Path(output_root) / result["created_at"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "desktop_vision_result.json"
    else:
        output_path = Path(fixed_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_image(output_root, color_bgr):
    """保存当前预览画面。"""

    import cv2

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "color.png"
    cv2.imwrite(str(output_path), color_bgr)
    return output_path


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="YOLO .engine 或 .pt 路径")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="YOLO 输入尺寸")
    parser.add_argument("--show", action="store_true", help="打开实时预览窗口")
    parser.add_argument("--protocol", action="store_true", help="单帧模式下打印 @OBJ/@GRASP 协议")
    args = parser.parse_args(argv)

    if args.show:
        run_live(model_path=args.model, output_root=args.output_root, conf=args.conf, imgsz=args.imgsz)
    else:
        run_once(
            model_path=args.model,
            output_root=args.output_root,
            conf=args.conf,
            imgsz=args.imgsz,
            protocol=args.protocol,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
