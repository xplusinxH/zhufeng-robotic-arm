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
from apriltag.pose_sample import robust_average_transforms
from camera.realsense_camera import RealSenseCamera
from communication.tag_pose_protocol import (
    format_error,
    format_invalid_pose_sample,
    format_pose_sample,
    is_get_tag_pose_command,
    is_get_tool_command,
)
from coordinate.pose_transform import relative_transform


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
    frame_limit=None,
):
    """Run camera detection and respond to serial 6D pose sample queries."""
    import numpy as np
    import serial

    camera = RealSenseCamera()
    detector = AprilTagPoseDetector()
    serial_device = serial.Serial(serial_port, baudrate=baudrate, timeout=0)
    receive_buffer = ""
    handled_queries = 0

    camera.start()
    try:
        camera_params = _intrinsics_to_camera_params(camera.get_color_intrinsics())
        print(
            "AprilTag 6D 位姿串口服务已启动：port={} baudrate={} tag_size_m={:.4f} base_id={} tool_id={}".format(
                serial_port, baudrate, tag_size_m, base_tag_id, tool_tag_id
            )
        )
        while frame_limit is None or handled_queries < frame_limit:
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
):
    data = serial_device.read(256)
    if data:
        receive_buffer += data.decode("ascii", "ignore")

    query_count = 0
    while "#" in receive_buffer:
        frame, receive_buffer = receive_buffer.split("#", 1)
        message = frame + "#"
        if is_get_tag_pose_command(message):
            response = _capture_pose_sample_json(
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
            )
            serial_device.write((response + "\n").encode("utf-8"))
            print("serial <= {}  => {}".format(message, response))
            query_count += 1
        elif is_get_tool_command(message):
            response = format_error("deprecated command; use @GET_TAG_POSE#")
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
):
    valid_relative_transforms = []
    base_ref_seen = False
    tool0_seen = False

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
        if base_tag_id in detections and tool_tag_id in detections:
            valid_relative_transforms.append(
                relative_transform(detections[base_tag_id], detections[tool_tag_id])
            )

    timestamp = _timestamp_now()
    sample_id = "S{:04d}".format(seq)
    if len(valid_relative_transforms) < min_valid_frames:
        return format_invalid_pose_sample(
            sample_id=sample_id,
            seq=seq,
            timestamp_jetson=timestamp,
            tag_size_m=tag_size_m,
            frame_count_used=len(valid_relative_transforms),
            base_ref_seen=base_ref_seen,
            tool0_seen=tool0_seen,
            tag_base_ref_id=base_tag_id,
            tag_tool0_id=tool_tag_id,
        )

    fused_transform = robust_average_transforms(valid_relative_transforms)
    return format_pose_sample(
        sample_id=sample_id,
        seq=seq,
        timestamp_jetson=timestamp,
        transform=fused_transform,
        tag_size_m=tag_size_m,
        frame_count_used=len(valid_relative_transforms),
        base_ref_seen=base_ref_seen,
        tool0_seen=tool0_seen,
        tag_base_ref_id=base_tag_id,
        tag_tool0_id=tool_tag_id,
    )


def _timestamp_now():
    return datetime.now().isoformat(timespec="milliseconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")
    parser.add_argument("--tag-size-mm", type=float, required=True, help="AprilTag 黑白图案边长，单位毫米")
    parser.add_argument("--base-tag-id", type=int, default=1, help="底座参考 tag id")
    parser.add_argument("--tool-tag-id", type=int, default=0, help="末端 tool0 tag id")
    parser.add_argument("--sample-frames", type=int, default=15, help="每个 sample 连续采集帧数")
    parser.add_argument("--min-valid-frames", type=int, default=5, help="最少有效双 tag 帧数")
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
        frame_limit=args.frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
