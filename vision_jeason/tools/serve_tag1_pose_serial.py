"""Serve AprilTag 6D pose samples over Jetson serial."""

import argparse
from datetime import datetime
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apriltag.detector import AprilTagPoseDetector
from apriltag.pose_sample import BaseReferenceCache, robust_average_transforms
from camera.realsense_camera import RealSenseCamera
from communication.tag_pose_protocol import (
    format_error,
    format_invalid_pose_sample,
    format_pose_sample,
    is_get_tag_pose_command,
    is_get_tool_command,
)
from coordinate.pose_transform import relative_transform
from tools.tag_debug_view import draw_debug_overlay, should_quit_from_key


def _intrinsics_to_camera_params(intrinsics):
    return (
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy),
    )


def run(
    serial_port,
    baudrate,
    tag_size_m,
    base_tag_id,
    tool_tag_id,
    sample_frames,
    min_valid_frames,
    base_cache_items,
    show=False,
    frame_limit=None,
):
    """Run camera detection and respond to serial 6D pose sample queries."""
    import cv2
    import numpy as np
    import serial

    camera = RealSenseCamera()
    detector = AprilTagPoseDetector()
    base_ref_cache = BaseReferenceCache(max_items=base_cache_items)
    serial_device = serial.Serial(serial_port, baudrate=baudrate, timeout=0)
    receive_buffer = ""
    handled_queries = 0
    debug_state = {"base_ref_source": "none", "last_status": "waiting"}

    camera.start()
    try:
        camera_params = _intrinsics_to_camera_params(camera.get_color_intrinsics())
        print(
            "AprilTag 6D 位姿串口服务已启动：port={} baudrate={} tag_size_m={:.4f} base_id={} tool_id={} show={}".format(
                serial_port, baudrate, tag_size_m, base_tag_id, tool_tag_id, show
            )
        )
        while frame_limit is None or handled_queries < frame_limit:
            if show and _update_debug_window(
                camera=camera,
                detector=detector,
                camera_params=camera_params,
                tag_size_m=tag_size_m,
                base_tag_id=base_tag_id,
                tool_tag_id=tool_tag_id,
                np_module=np,
                cv2_module=cv2,
                debug_state=debug_state,
            ):
                break

            receive_buffer, query_count = _handle_serial_queries(
                serial_device=serial_device,
                receive_buffer=receive_buffer,
                camera=camera,
                detector=detector,
                camera_params=camera_params,
                tag_size_m=tag_size_m,
                base_tag_id=base_tag_id,
                tool_tag_id=tool_tag_id,
                sample_frames=sample_frames,
                min_valid_frames=min_valid_frames,
                next_seq=handled_queries + 1,
                np_module=np,
                base_ref_cache=base_ref_cache,
                debug_state=debug_state,
            )
            handled_queries += query_count
            time.sleep(0.001)
    except Exception as exc:
        try:
            serial_device.write((format_error(str(exc)) + "\n").encode("ascii", "ignore"))
        finally:
            raise
    finally:
        camera.stop()
        serial_device.close()
        if show:
            cv2.destroyAllWindows()


def _update_debug_window(
    camera,
    detector,
    camera_params,
    tag_size_m,
    base_tag_id,
    tool_tag_id,
    np_module,
    cv2_module,
    debug_state,
):
    color_frame, _depth_frame = camera.capture_aligned()
    color_bgr = np_module.asanyarray(color_frame.get_data()).copy()
    detections = detector.detect(
        color_bgr,
        camera_params=camera_params,
        tag_size_m=tag_size_m,
    )
    draw_debug_overlay(
        image_bgr=color_bgr,
        detections=detections,
        base_tag_id=base_tag_id,
        tool_tag_id=tool_tag_id,
        base_ref_source=debug_state.get("base_ref_source", "none"),
        last_status=debug_state.get("last_status", "waiting"),
    )
    cv2_module.imshow("Sukinee AprilTag Debug", color_bgr)
    return should_quit_from_key(cv2_module.waitKey(1))


def _handle_serial_queries(
    serial_device,
    receive_buffer,
    camera,
    detector,
    camera_params,
    tag_size_m,
    base_tag_id,
    tool_tag_id,
    sample_frames,
    min_valid_frames,
    next_seq,
    np_module,
    base_ref_cache,
    debug_state,
):
    data = serial_device.read(256)
    if data:
        receive_buffer += data.decode("ascii", "ignore")

    query_count = 0
    while "#" in receive_buffer:
        frame, receive_buffer = receive_buffer.split("#", 1)
        message = frame + "#"
        if is_get_tag_pose_command(message):
            response, status = _capture_pose_sample_json(
                camera=camera,
                detector=detector,
                camera_params=camera_params,
                tag_size_m=tag_size_m,
                base_tag_id=base_tag_id,
                tool_tag_id=tool_tag_id,
                sample_frames=sample_frames,
                min_valid_frames=min_valid_frames,
                seq=next_seq + query_count,
                np_module=np_module,
                base_ref_cache=base_ref_cache,
            )
            debug_state["base_ref_source"] = status["base_ref_source"]
            debug_state["last_status"] = status["last_status"]
            serial_device.write((response + "\n").encode("utf-8"))
            print("serial <= {}  => {}".format(message, response))
            query_count += 1
        elif is_get_tool_command(message):
            response = format_error("deprecated command; use @GET_TAG_POSE#")
            debug_state["last_status"] = "deprecated command"
            serial_device.write((response + "\n").encode("ascii", "ignore"))
    return receive_buffer[-128:], query_count


def _capture_pose_sample_json(
    camera,
    detector,
    camera_params,
    tag_size_m,
    base_tag_id,
    tool_tag_id,
    sample_frames,
    min_valid_frames,
    seq,
    np_module,
    base_ref_cache,
):
    valid_relative_transforms = []
    base_ref_seen = False
    tool0_seen = False
    used_cached_base = False
    used_live_base = False

    for _index in range(sample_frames):
        color_frame, _depth_frame = camera.capture_aligned()
        color_bgr = np_module.asanyarray(color_frame.get_data())
        detections = detector.detect_camera_to_tag(
            color_bgr,
            camera_params=camera_params,
            tag_size_m=tag_size_m,
        )
        base_ref_seen = base_ref_seen or base_tag_id in detections
        tool0_seen = tool0_seen or tool_tag_id in detections

        if base_tag_id in detections:
            base_ref_cache.add(detections[base_tag_id])

        if tool_tag_id not in detections:
            continue

        camera_to_base = None
        if base_tag_id in detections:
            camera_to_base = detections[base_tag_id]
            used_live_base = True
        elif base_ref_cache.has_value():
            camera_to_base = base_ref_cache.get_fused()
            used_cached_base = True

        if camera_to_base is not None:
            valid_relative_transforms.append(
                relative_transform(camera_to_base, detections[tool_tag_id])
            )

    timestamp = _timestamp_now()
    sample_id = "S{:04d}".format(seq)
    base_ref_source = _base_ref_source(used_live_base, used_cached_base)
    if len(valid_relative_transforms) < min_valid_frames:
        return (
            format_invalid_pose_sample(
                sample_id=sample_id,
                seq=seq,
                timestamp_jetson=timestamp,
                tag_size_m=tag_size_m,
                frame_count_used=len(valid_relative_transforms),
                base_ref_seen=base_ref_seen,
                tool0_seen=tool0_seen,
                base_ref_source=base_ref_source,
                tag_base_ref_id=base_tag_id,
                tag_tool0_id=tool_tag_id,
            ),
            {"base_ref_source": base_ref_source, "last_status": "invalid sample"},
        )

    fused_transform = robust_average_transforms(valid_relative_transforms)
    return (
        format_pose_sample(
            sample_id=sample_id,
            seq=seq,
            timestamp_jetson=timestamp,
            transform=fused_transform,
            tag_size_m=tag_size_m,
            frame_count_used=len(valid_relative_transforms),
            base_ref_seen=base_ref_seen,
            tool0_seen=tool0_seen,
            base_ref_source=base_ref_source,
            tag_base_ref_id=base_tag_id,
            tag_tool0_id=tool_tag_id,
        ),
        {"base_ref_source": base_ref_source, "last_status": "ok"},
    )


def _base_ref_source(used_live_base, used_cached_base):
    if used_live_base and used_cached_base:
        return "mixed"
    if used_live_base:
        return "live"
    if used_cached_base:
        return "cached"
    return "none"


def _timestamp_now():
    return datetime.now().isoformat(timespec="milliseconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")
    parser.add_argument("--tag-size-mm", type=float, required=True, help="AprilTag 黑色外框边长，单位毫米")
    parser.add_argument("--base-tag-id", type=int, default=0, help="底座参考 tag id")
    parser.add_argument("--tool-tag-id", type=int, default=1, help="末端 tool0 tag id")
    parser.add_argument("--sample-frames", type=int, default=15, help="每个 sample 连续采集帧数")
    parser.add_argument("--min-valid-frames", type=int, default=5, help="最少有效 tool0 帧数")
    parser.add_argument("--base-cache-items", type=int, default=20, help="缓存的底座参考观测数量")
    parser.add_argument("--show", action="store_true", help="在 Jetson 屏幕显示 AprilTag 调试窗口")
    parser.add_argument("--frames", type=int, help="测试用：处理指定查询次数后退出")
    args = parser.parse_args()

    run(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        tag_size_m=args.tag_size_mm / 1000.0,
        base_tag_id=args.base_tag_id,
        tool_tag_id=args.tool_tag_id,
        sample_frames=args.sample_frames,
        min_valid_frames=args.min_valid_frames,
        base_cache_items=args.base_cache_items,
        show=args.show,
        frame_limit=args.frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
