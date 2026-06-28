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
from tools.eye_in_hand_debug_view import (
    WINDOW_NAME,
    draw_eye_in_hand_debug_overlay,
    should_quit_from_key,
)
from tools.offline_eye_in_hand_debug import build_offline_eye_in_hand_result, parse_roi

DEFAULT_OUTPUT_ROOT = Path("/mnt/zhufeng_data/data/eye_in_hand_debug")
DEFAULT_GRIPPER_MAX_OPENING_M = 0.08


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
    camera_keepout_roi=None,
    min_visibility=0.60,
    segmentation_mode="table_plane",
    enable_table_z_compensation=False,
    gripper_max_opening_m=DEFAULT_GRIPPER_MAX_OPENING_M,
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
            image_size=(len(depth_m[0]) if depth_m else 0, len(depth_m)),
            camera_keepout_roi=camera_keepout_roi,
            min_visibility=min_visibility,
            segmentation_mode=segmentation_mode,
            enable_table_z_compensation=enable_table_z_compensation,
            gripper_max_opening_m=gripper_max_opening_m,
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


def run_eye_in_hand_live_view(
    camera,
    base_from_tool,
    tool_from_camera,
    output_root,
    min_points=20,
    pixel_radius=3,
    stride=1,
    min_z_base_m=0.01,
    max_z_base_m=0.30,
    camera_keepout_roi=None,
    min_visibility=0.60,
    segmentation_mode="table_plane",
    enable_table_z_compensation=False,
    gripper_max_opening_m=DEFAULT_GRIPPER_MAX_OPENING_M,
):
    """打开实时调试窗口，优先保证画面流畅显示。

    Jetson Nano 算力有限，不能每帧都跑完整 OBJ/GRASP 检测，否则窗口会像卡死。
    因此实时模式默认只显示相机画面，按 ``D`` 手动检测当前帧，按 ``S`` 保存当前显示结果。
    """

    import cv2
    import numpy as np

    camera.start()
    try:
        intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
        depth_scale = camera.get_depth_scale()
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        print("实时调试窗口已打开：按 D 检测当前帧，按 S 保存当前画面，按 Q 或 Esc 退出。")
        print("调试输出目录：{0}".format(output_root))
        latest_result = make_empty_live_result(intrinsics, (camera.width, camera.height))
        latest_overlay = None
        latest_detection_color_bgr = None
        latest_detection_depth_m = None
        while True:
            color_frame, depth_frame = camera.capture_aligned()
            color_bgr = np.asanyarray(color_frame.get_data())
            overlay = draw_eye_in_hand_debug_overlay(
                color_bgr,
                latest_result,
                camera_keepout_roi=camera_keepout_roi,
                cv2_module=cv2,
            )
            cv2.putText(
                overlay,
                "D=detect  S=save  Q/Esc=quit",
                (10, max(44, camera.height - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            latest_overlay = overlay
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(1) & 0xFF
            if should_quit_from_key(key):
                break
            if key in (ord("d"), ord("D")):
                depth_m = raw_depth_to_depth_m(
                    np.asanyarray(depth_frame.get_data()),
                    depth_scale,
                )
                latest_detection_color_bgr = color_bgr.copy()
                latest_detection_depth_m = depth_m
                latest_result = build_offline_eye_in_hand_result(
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    base_from_tool=base_from_tool,
                    tool_from_camera=tool_from_camera,
                    min_points=min_points,
                    pixel_radius=pixel_radius,
                    stride=stride,
                    min_z_base_m=min_z_base_m,
                    max_z_base_m=max_z_base_m,
                    image_size=(len(depth_m[0]) if depth_m else 0, len(depth_m)),
                    camera_keepout_roi=camera_keepout_roi,
                    min_visibility=min_visibility,
                    segmentation_mode=segmentation_mode,
                    enable_table_z_compensation=enable_table_z_compensation,
                    gripper_max_opening_m=gripper_max_opening_m,
                )
                _attach_capture_metadata(latest_result, intrinsics, depth_m)
                _print_table_plane_diagnostics(latest_result.get("table_plane"))
                print(
                    "当前帧检测完成：OBJ={0} GRASP={1} REJECT={2}".format(
                        latest_result["candidate_count"],
                        latest_result["grasp_count"],
                        latest_result["rejected_grasp_count"],
                    )
                )
            if key in (ord("s"), ord("S")):
                output_path = save_capture_result(
                    output_root,
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    latest_result,
                )
                latest_result["output_path"] = str(output_path)
                latest_result["debug_artifacts"] = save_capture_debug_artifacts(
                    output_path.parent,
                    overlay_bgr=latest_overlay,
                    current_color_bgr=color_bgr,
                    detection_color_bgr=latest_detection_color_bgr,
                    detection_depth_m=latest_detection_depth_m,
                    cv2_module=cv2,
                    np_module=np,
                )
                _rewrite_result_with_output_path(output_path, latest_result)
                print("当前帧调试结果已保存到：{0}".format(output_path.parent))
    finally:
        camera.stop()
        cv2.destroyAllWindows()


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


def raw_depth_to_depth_m(depth_raw, depth_scale):
    """将 RealSense 原始深度缓冲区批量转换为米单位矩阵。

    实时窗口不能逐像素调用 ``get_distance``，否则 Jetson Nano 第一帧就会明显卡顿。
    真机运行时通常传入 numpy 数组，单元测试也允许传入普通二维列表。
    """

    if hasattr(depth_raw, "astype"):
        depth_m = depth_raw.astype("float32") * float(depth_scale)
        return depth_m.tolist()
    return [
        [float(value) * float(depth_scale) for value in row]
        for row in depth_raw
    ]


def make_empty_live_result(intrinsics, image_size):
    """构造实时窗口初始结果。

    用户未按 ``D`` 触发检测前，窗口仍然需要可以保存当前画面。这个空结果
    明确表示“尚未检测到候选物”，而不是检测失败。
    """

    width, height = image_size
    return {
        "frame": "base",
        "base_from_camera": None,
        "candidate_count": 0,
        "candidates": [],
        "grasp_count": 0,
        "rejected_grasp_count": 0,
        "grasps": [],
        "intrinsics": intrinsics,
        "depth_size": {
            "width": int(width),
            "height": int(height),
        },
        "captured_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "source": "live_preview_no_detection",
    }


def save_capture_result(output_root, timestamp, result):
    """将本次单帧调试结果保存为 JSON。"""

    capture_dir = Path(output_root) / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)
    output_path = capture_dir / "eye_in_hand_candidates.json"
    _rewrite_result_with_output_path(output_path, result)
    return output_path


def save_capture_debug_artifacts(
    capture_dir,
    overlay_bgr,
    current_color_bgr,
    detection_color_bgr,
    detection_depth_m,
    cv2_module,
    np_module,
):
    """保存漏检复盘需要的原始图像和深度数据。"""

    capture_dir = Path(capture_dir)
    artifacts = {}
    if overlay_bgr is not None:
        overlay_path = capture_dir / "overlay.png"
        cv2_module.imwrite(str(overlay_path), overlay_bgr)
        artifacts["overlay_png"] = overlay_path.name
    color_image = detection_color_bgr if detection_color_bgr is not None else current_color_bgr
    if color_image is not None:
        color_path = capture_dir / "color.png"
        cv2_module.imwrite(str(color_path), color_image)
        artifacts["color_png"] = color_path.name
    if detection_depth_m is not None:
        depth_array = np_module.asarray(detection_depth_m, dtype="float32")
        depth_path = capture_dir / "depth_m.npy"
        np_module.save(str(depth_path), depth_array)
        artifacts["depth_m_npy"] = depth_path.name
        preview_path = capture_dir / "depth_preview.png"
        cv2_module.imwrite(str(preview_path), _make_depth_preview(depth_array, np_module))
        artifacts["depth_preview_png"] = preview_path.name
    return artifacts


def _make_depth_preview(depth_m, np_module):
    """把米单位深度矩阵转换成 8-bit 预览图，便于直接查看。"""

    valid = depth_m[depth_m > 0.0]
    if valid.size == 0:
        return np_module.zeros(depth_m.shape, dtype="uint8")
    min_depth = float(valid.min())
    max_depth = float(valid.max())
    if max_depth <= min_depth:
        return np_module.zeros(depth_m.shape, dtype="uint8")
    normalized = (depth_m - min_depth) * (255.0 / (max_depth - min_depth))
    normalized = np_module.clip(normalized, 0, 255)
    normalized[depth_m <= 0.0] = 0
    return normalized.astype("uint8")


def _attach_capture_metadata(result, intrinsics, depth_m):
    """补充实时/单帧调试结果需要携带的相机元数据。"""

    result["intrinsics"] = intrinsics
    result["depth_size"] = {
        "width": len(depth_m[0]) if depth_m else 0,
        "height": len(depth_m),
    }
    result["captured_at"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _print_table_plane_diagnostics(table_plane):
    """在实时调试终端输出桌面平面反验结果，便于现场快速判断 Z 轴误差。"""

    if not table_plane or not table_plane.get("valid"):
        reason = table_plane.get("reason", "unknown") if table_plane else "missing"
        print("桌面平面反验无效：{0}".format(reason))
        return
    print(
        "桌面平面反验：z_offset={0:.3f}m z_comp={1:.3f}m tilt={2:.2f}deg rmse={3:.4f}m".format(
            float(table_plane["table_z_offset_m"]),
            float(table_plane["z_compensation_m"]),
            float(table_plane["table_tilt_deg"]),
            float(table_plane["fit_rmse_m"]),
        )
    )


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
    parser.add_argument("--camera-keepout-roi", help="相机视野禁区 ROI，格式 u1,v1,u2,v2")
    parser.add_argument("--min-visibility", type=float, default=0.60, help="生成 GRASP 的最小视野评分")
    parser.add_argument("--show", action="store_true", help="打开实时可视化窗口")
    parser.add_argument(
        "--table-z-compensation",
        action="store_true",
        help="使用拟合桌面平面对候选物和 GRASP 的 Z_base 做临时补偿",
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=("table_plane", "depth_foreground", "base_height"),
        default="table_plane",
        help="候选物分割模式；现场调试默认使用 table_plane",
    )
    args = parser.parse_args(argv)

    tool_camera = load_tool_camera_record(args.tool_camera)
    base_tool_pose = load_base_tool_pose_from_file(args.base_tool_pose)
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)
    if args.show:
        run_eye_in_hand_live_view(
            camera=camera,
            base_from_tool=base_tool_pose["transform"],
            tool_from_camera=tool_camera["transform"],
            output_root=args.output_root,
            min_points=args.min_points,
            pixel_radius=max(args.pixel_radius, max(args.stride, 8)),
            stride=max(args.stride, 8),
            min_z_base_m=args.min_z_base,
            max_z_base_m=args.max_z_base,
            camera_keepout_roi=parse_roi(args.camera_keepout_roi),
            min_visibility=args.min_visibility,
            segmentation_mode=args.segmentation_mode,
            enable_table_z_compensation=args.table_z_compensation,
        )
        return 0

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
        camera_keepout_roi=parse_roi(args.camera_keepout_roi),
        min_visibility=args.min_visibility,
        segmentation_mode=args.segmentation_mode,
        enable_table_z_compensation=args.table_z_compensation,
    )
    print("单帧 eye-in-hand 调试结果已保存到：{}".format(result["output_path"]))
    print("候选物数量：{}".format(result["candidate_count"]))
    print("抓取建议数量：{}".format(result["grasp_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
