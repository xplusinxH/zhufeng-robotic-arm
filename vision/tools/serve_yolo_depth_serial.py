"""Jetson YOLO 深度识别串口服务。

控制端发送 ``@DETECT#`` 后，本服务采集一帧 D435 图像，执行 YOLO/TensorRT
识别和深度 ROI 3D 几何计算，然后返回现有 ``@OBJ``、``@GRASP``、``@END`` 协议帧。
服务不主动连续识别，避免在机械臂运动过程中无意义地占满 Jetson 算力。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from communication.protocol import format_error
from communication.pose_protocol import is_pose_frame, parse_pose_frame
from tools.run_yolo_depth_scene import format_scene_protocol_frames, run_single_scene


DETECT_COMMAND = "@DETECT#"
POSE_OK_FRAME = "@POSE_OK#"


@dataclass
class YoloDepthSerialState:
    """串口服务运行状态，保存控制端最近一次发送的 ``T_base_tool``。"""

    base_from_tool: object = None


def is_detect_command(message):
    """判断串口帧是否为一次识别请求。"""

    return str(message).strip() == DETECT_COMMAND


def handle_serial_receive_buffer(receive_buffer, incoming_text, capture_scene, state=None):
    """处理串口接收缓冲区，返回剩余半包和待发送响应帧。

    ``capture_scene`` 是无参数回调，返回已经格式化好的协议帧列表。测试中用假函数，
    真机运行时绑定到 YOLO 深度识别。
    """

    if state is None:
        state = YoloDepthSerialState()
    receive_buffer = (receive_buffer or "") + (incoming_text or "")
    responses = []
    while "#" in receive_buffer:
        frame_body, receive_buffer = receive_buffer.split("#", 1)
        message = frame_body + "#"
        if is_pose_frame(message):
            try:
                state.base_from_tool = parse_pose_frame(message)["transform"]
                responses.append(POSE_OK_FRAME)
            except Exception as exc:
                responses.append(format_error("BAD_POSE", str(exc)))
        elif is_detect_command(message):
            if state.base_from_tool is None:
                responses.append(format_error("NO_POSE", "send_POSE_before_DETECT"))
                continue
            try:
                responses.extend(capture_scene())
            except Exception as exc:
                responses.append(format_error("DETECT_FAILED", str(exc)))
        else:
            responses.append(format_error("BAD_COMMAND", "use DETECT"))
    return receive_buffer[-128:], responses


def run_service(
    serial_port="/dev/ttyUSB0",
    baudrate=115200,
    model_path=None,
    conf=0.50,
    imgsz=640,
    depth_stride=2,
    min_depth_points=20,
    poll_sleep_s=0.001,
):
    """启动常驻串口服务。"""

    import serial

    serial_device = serial.Serial(serial_port, baudrate=baudrate, timeout=0)
    receive_buffer = ""
    state = YoloDepthSerialState()

    def capture_scene_frames():
        result = run_single_scene(
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            depth_stride=depth_stride,
            min_depth_points=min_depth_points,
            print_protocol=False,
            base_from_tool=state.base_from_tool,
        )
        return format_scene_protocol_frames(result)

    print(
        "YOLO 深度识别串口服务已启动：port={0} baudrate={1} command={2}".format(
            serial_port,
            baudrate,
            DETECT_COMMAND,
        )
    )
    try:
        while True:
            data = serial_device.read(256)
            if data:
                receive_buffer, responses = handle_serial_receive_buffer(
                    receive_buffer=receive_buffer,
                    incoming_text=data.decode("ascii", "ignore"),
                    capture_scene=capture_scene_frames,
                    state=state,
                )
                for response in responses:
                    serial_device.write((response + "\n").encode("ascii", "ignore"))
                    print("serial => {0}".format(response))
            time.sleep(float(poll_sleep_s))
    finally:
        serial_device.close()


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")
    parser.add_argument("--model", type=Path, help="YOLO .engine 或 .pt 路径")
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--depth-stride", type=int, default=2, help="ROI 深度采样步长")
    parser.add_argument("--min-depth-points", type=int, default=20, help="ROI 最少有效深度点")
    args = parser.parse_args(argv)

    run_service(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        depth_stride=args.depth_stride,
        min_depth_points=args.min_depth_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
