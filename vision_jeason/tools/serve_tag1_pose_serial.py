"""Serve AprilTag end-effector coordinates over Jetson serial."""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apriltag.detector import AprilTagPoseDetector
from apriltag.pose_service import TagPoseService
from camera.realsense_camera import RealSenseCamera
from communication.tag_pose_protocol import format_error, is_get_tool_command


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
    max_age_s,
    frame_limit=None,
):
    """Run camera detection and respond to serial coordinate queries."""
    import numpy as np
    import serial

    camera = RealSenseCamera()
    detector = AprilTagPoseDetector()
    service = TagPoseService(
        base_tag_id=base_tag_id,
        tool_tag_id=tool_tag_id,
        max_age_s=max_age_s,
    )
    serial_device = serial.Serial(serial_port, baudrate=baudrate, timeout=0)
    receive_buffer = ""
    frame_count = 0

    camera.start()
    try:
        camera_params = _intrinsics_to_camera_params(camera.get_color_intrinsics())
        print(
            "AprilTag 串口服务已启动：port={} baudrate={} tag_size_m={:.4f}".format(
                serial_port, baudrate, tag_size_m
            )
        )
        while frame_limit is None or frame_count < frame_limit:
            color_frame, _depth_frame = camera.capture_aligned()
            color_bgr = np.asanyarray(color_frame.get_data())
            detections = detector.detect_camera_to_tag(
                color_bgr,
                camera_params=camera_params,
                tag_size_m=tag_size_m,
            )
            updated = service.update_from_detections(detections)
            frame_count += 1

            if updated:
                cached = service.get_cached()
                if cached is not None:
                    position_m, age_ms = cached
                    print(
                        "tag1 in tag0: x={:.1f} y={:.1f} z={:.1f} mm age={} ms".format(
                            position_m[0] * 1000.0,
                            position_m[1] * 1000.0,
                            position_m[2] * 1000.0,
                            age_ms,
                        )
                    )

            receive_buffer = _handle_serial_queries(serial_device, receive_buffer, service)
            time.sleep(0.001)
    except Exception as exc:
        try:
            serial_device.write((format_error(str(exc)) + "\n").encode("ascii", "ignore"))
        finally:
            raise
    finally:
        camera.stop()
        serial_device.close()


def _handle_serial_queries(serial_device, receive_buffer, service):
    data = serial_device.read(256)
    if data:
        receive_buffer += data.decode("ascii", "ignore")

    while "#" in receive_buffer:
        frame, receive_buffer = receive_buffer.split("#", 1)
        message = frame + "#"
        if is_get_tool_command(message):
            response = service.format_response()
            serial_device.write((response + "\n").encode("ascii"))
            print("serial <= {}  => {}".format(message, response))
    return receive_buffer[-128:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")
    parser.add_argument("--tag-size-mm", type=float, required=True, help="AprilTag 黑白图案边长，单位毫米")
    parser.add_argument("--base-tag-id", type=int, default=0, help="底座 tag id")
    parser.add_argument("--tool-tag-id", type=int, default=1, help="末端 tag id")
    parser.add_argument("--max-age-ms", type=int, default=500, help="坐标缓存有效期，单位毫秒")
    parser.add_argument("--frames", type=int, help="测试用：处理指定帧数后退出")
    args = parser.parse_args()

    run(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        tag_size_m=args.tag_size_mm / 1000.0,
        base_tag_id=args.base_tag_id,
        tool_tag_id=args.tool_tag_id,
        max_age_s=args.max_age_ms / 1000.0,
        frame_limit=args.frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
