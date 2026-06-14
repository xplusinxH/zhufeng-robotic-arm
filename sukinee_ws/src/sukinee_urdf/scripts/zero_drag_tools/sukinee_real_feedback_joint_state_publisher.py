#!/usr/bin/env python3
import argparse
import json
import re
import select
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


CAN_IFACE_DEFAULT = "can0"
HOST_ID = 0xFD
COMM_READ_PARAM = 17

READ_TIMEOUT_SEC = 2.0
MAX_READ_ATTEMPTS = 2
RETRY_DELAY_SEC = 0.05
INTER_PARAM_DELAY_SEC = 0.01
INTER_MOTOR_DELAY_SEC = 0.005

ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

PARAMS = [
    (0x7019, "pos", "rad"),
    (0x701A, "iqf", "A"),
    (0x701B, "vel", "rad/s"),
    (0x701C, "vbus", "V"),
]

JOINT_NAME_BY_ID = {
    1: "Joint1",
    2: "Joint2",
    3: "Joint3",
    4: "Joint4",
    5: "Joint5",
    6: "Joint6",
}

# RViz sanity limits only. These do NOT command or clamp the real motor.
URDF_WARN_LIMITS = {
    1: (-12.57, 12.57),
    2: (-12.57, 12.57),
    3: (-12.57, 12.57),
    4: (-12.57, 12.57),
    5: (-12.57, 12.57),
    6: (-0.79, 0.79),   # Current project state: Joint6 is finite revolute.
}


def build_extended_can_id(comm_type: int, data_area2: int, target_id: int) -> int:
    """
    RobStride private protocol 29-bit CAN ID:
      Bit28~24: comm_type
      Bit23~8 : data_area2
      Bit7~0  : target_id
    """
    return ((comm_type & 0x1F) << 24) | ((data_area2 & 0xFFFF) << 8) | (target_id & 0xFF)


def make_read_frame(motor_id: int, index: int) -> str:
    can_id = build_extended_can_id(COMM_READ_PARAM, HOST_ID, motor_id)

    index_lo = index & 0xFF
    index_hi = (index >> 8) & 0xFF

    data = f"{index_lo:02X}{index_hi:02X}000000000000"
    return f"{can_id:08X}#{data}"


def send_can_frame(can_iface: str, frame: str) -> None:
    subprocess.run(
        ["cansend", can_iface, frame],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def send_read(can_iface: str, motor_id: int, index: int) -> None:
    send_can_frame(can_iface, make_read_frame(motor_id, index))


def parse_float_from_parts(parts) -> float:
    if len(parts) != 8:
        raise ValueError(f"Expected 8 bytes, got: {' '.join(parts)}")
    raw = bytes(int(x, 16) for x in parts[4:8])
    return struct.unpack("<f", raw)[0]


def parse_candump_line(line: str):
    if "[8]" not in line:
        return None

    left, data_hex = line.split("[8]", 1)
    parts = data_hex.strip().split()

    if len(parts) != 8:
        return None

    can_id = None
    for token in reversed(left.strip().split()):
        token = token.strip().upper()
        if re.fullmatch(r"[0-9A-F]{3,8}", token):
            can_id = token
            break

    if can_id is None:
        return None

    return can_id, parts


class CandumpReader:
    def __init__(self, can_iface: str):
        self.can_iface = can_iface
        self.proc = subprocess.Popen(
            ["candump", "-td", can_iface],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        time.sleep(0.10)

        if self.proc.poll() is not None:
            _, err = self.proc.communicate()
            raise RuntimeError(f"candump exited early: {err.strip()}")

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=1.0)

    def drain_available(self):
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("candump process exited unexpectedly.")

            ready, _, _ = select.select([self.proc.stdout], [], [], 0.0)
            if not ready:
                return

            line = self.proc.stdout.readline()
            if not line:
                return

    def readline_until(self, deadline: float):
        if self.proc.poll() is not None:
            raise RuntimeError("candump process exited unexpectedly.")

        timeout = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)

        if not ready:
            return None

        return self.proc.stdout.readline()


def read_param(reader: CandumpReader, can_iface: str, motor_id: int, index: int):
    success_resp_id = f"1100{motor_id:02X}FD"
    fail_resp_id = f"1101{motor_id:02X}FD"
    expected_index = f"{index & 0xFF:02X}{(index >> 8) & 0xFF:02X}"

    for attempt in range(1, MAX_READ_ATTEMPTS + 1):
        reader.drain_available()

        try:
            send_read(can_iface, motor_id, index)
        except subprocess.CalledProcessError:
            return "SEND_ERROR", None

        deadline = time.monotonic() + READ_TIMEOUT_SEC

        while time.monotonic() < deadline:
            line = reader.readline_until(deadline)
            if line is None:
                break

            parsed = parse_candump_line(line)
            if parsed is None:
                continue

            can_id, parts = parsed

            if can_id not in (success_resp_id, fail_resp_id):
                continue

            rx_index = (parts[0] + parts[1]).upper()
            if rx_index != expected_index:
                continue

            if can_id == fail_resp_id:
                return "READ_FAIL", None

            try:
                value = parse_float_from_parts(parts)
            except Exception:
                return "PARSE_ERROR", None

            return "OK", value

        if attempt < MAX_READ_ATTEMPTS:
            time.sleep(RETRY_DELAY_SEC)

    return "TIMEOUT", None


def read_all_feedback(can_iface: str):
    values: Dict[int, Dict[str, Optional[float]]] = {}
    statuses: Dict[int, Dict[str, str]] = {}

    reader = CandumpReader(can_iface)

    try:
        for motor_id in ALL_MOTOR_IDS:
            values[motor_id] = {}
            statuses[motor_id] = {}

            for index, name, _unit in PARAMS:
                status, value = read_param(reader, can_iface, motor_id, index)
                statuses[motor_id][name] = status
                values[motor_id][name] = value

                time.sleep(INTER_PARAM_DELAY_SEC)

            time.sleep(INTER_MOTOR_DELAY_SEC)
    finally:
        reader.close()

    return values, statuses


def feedback_ok(values, statuses) -> Tuple[bool, str]:
    for motor_id in ALL_MOTOR_IDS:
        for _, name, _ in PARAMS:
            status = statuses.get(motor_id, {}).get(name)
            if status != "OK":
                return False, f"Joint{motor_id} {name} status={status}"

        vbus = float(values[motor_id]["vbus"])
        iqf = float(values[motor_id]["iqf"])
        vel = float(values[motor_id]["vel"])

        if not (10.0 <= vbus <= 60.0):
            return False, f"Joint{motor_id} vbus out of range: {vbus:.3f} V"
        if abs(iqf) > 1.0:
            return False, f"Joint{motor_id} iqf too large for read-only sync: {iqf:.3f} A"
        if abs(vel) > 2.0:
            return False, f"Joint{motor_id} vel too large for safe visual sync: {vel:.3f} rad/s"

    return True, "OK"


def parse_joint_key_dict(raw: Dict[str, float]) -> Dict[int, float]:
    parsed: Dict[int, float] = {}

    for key, value in raw.items():
        if isinstance(key, str) and key.startswith("Joint"):
            motor_id = int(key.replace("Joint", ""))
        else:
            motor_id = int(key)

        parsed[motor_id] = float(value)

    return parsed


def load_offset_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "motor_to_urdf_sign" not in payload:
        raise RuntimeError("offset JSON missing key: motor_to_urdf_sign")
    if "motor_pos_at_urdf_zero" not in payload:
        raise RuntimeError("offset JSON missing key: motor_pos_at_urdf_zero")

    sign = parse_joint_key_dict(payload["motor_to_urdf_sign"])
    zero = parse_joint_key_dict(payload["motor_pos_at_urdf_zero"])

    for motor_id in ARM_JOINT_IDS:
        if motor_id not in sign:
            raise RuntimeError(f"offset JSON missing sign for Joint{motor_id}")
        if motor_id not in zero:
            raise RuntimeError(f"offset JSON missing zero for Joint{motor_id}")

    return sign, zero, payload


def map_motor_feedback_to_urdf_q(values, sign, zero) -> Dict[int, float]:
    q_map: Dict[int, float] = {}

    for motor_id in ARM_JOINT_IDS:
        motor_pos = float(values[motor_id]["pos"])
        q_map[motor_id] = sign[motor_id] * (motor_pos - zero[motor_id])

    return q_map


class SukineeRealFeedbackJointStatePublisher(Node):
    def __init__(self, args):
        super().__init__("sukinee_real_feedback_joint_state_publisher")

        self.args = args
        self.can_iface = args.can
        self.offset_json_path = Path(args.offset_json)
        self.sign, self.zero, self.offset_payload = load_offset_json(self.offset_json_path)

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.loop_count = 0
        self.timer = self.create_timer(1.0 / args.rate, self.timer_callback)

        self.print_startup_banner()

    def print_startup_banner(self):
        self.get_logger().info("Sukinee real feedback joint_state publisher")
        self.get_logger().info("Safety status:")
        self.get_logger().info("  NO Type1 motion command")
        self.get_logger().info("  NO Type3 enable")
        self.get_logger().info("  NO Type4 disable")
        self.get_logger().info("  NO Type18 parameter write")
        self.get_logger().info("  NO Type6 zero setting")
        self.get_logger().info("  NO save parameters")
        self.get_logger().info("  NO CAN_ID change")
        self.get_logger().info("  NO protocol change")
        self.get_logger().info("  NO MoveIt real execution")
        self.get_logger().info("  CAN usage is Type17 read-only parameter reads")
        self.get_logger().info(f"CAN interface: {self.can_iface}")
        self.get_logger().info(f"offset JSON: {self.offset_json_path}")

        self.get_logger().info("Loaded software mapping:")
        for motor_id in ARM_JOINT_IDS:
            self.get_logger().info(
                f"  Joint{motor_id}: sign={self.sign[motor_id]:+.1f}, "
                f"software_zero={self.zero[motor_id]:+.9f}"
            )

        self.get_logger().info(
            "Joint7 publishing mode: fixed 0.0 rad for now. "
            "Joint7 is not part of arm gravity offset calibration."
        )

    def timer_callback(self):
        self.loop_count += 1

        try:
            values, statuses = read_all_feedback(self.can_iface)
            ok, reason = feedback_ok(values, statuses)
            if not ok:
                self.get_logger().warn(f"Feedback check failed; not publishing this cycle: {reason}")
                return

            q_map = map_motor_feedback_to_urdf_q(values, self.sign, self.zero)

            for motor_id, q in q_map.items():
                low, high = URDF_WARN_LIMITS[motor_id]
                if not (low <= q <= high):
                    self.get_logger().warn(
                        f"Joint{motor_id} q_urdf={q:+.6f} rad outside expected URDF limit "
                        f"[{low:+.3f}, {high:+.3f}]"
                    )

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.name = [
                "Joint1",
                "Joint2",
                "Joint3",
                "Joint4",
                "Joint5",
                "Joint6",
                "Joint7",
            ]

            # Joint7 is published as 0.0 for now to keep gripper display stable.
            # Gripper-specific motor_pos -> URDF mapping can be calibrated later.
            msg.position = [
                float(q_map[1]),
                float(q_map[2]),
                float(q_map[3]),
                float(q_map[4]),
                float(q_map[5]),
                float(q_map[6]),
                0.0,
            ]

            # Velocity is optional, but useful for topic inspection.
            msg.velocity = [
                float(self.sign[1] * values[1]["vel"]),
                float(self.sign[2] * values[2]["vel"]),
                float(self.sign[3] * values[3]["vel"]),
                float(self.sign[4] * values[4]["vel"]),
                float(self.sign[5] * values[5]["vel"]),
                float(self.sign[6] * values[6]["vel"]),
                0.0,
            ]

            self.pub.publish(msg)

            if self.loop_count % self.args.print_every == 0:
                text = " | ".join(
                    [f"J{mid}={q_map[mid]:+.4f}" for mid in ARM_JOINT_IDS]
                )
                self.get_logger().info(f"Published /joint_states q_urdf: {text}")

        except Exception as e:
            self.get_logger().warn(f"Read/publish cycle failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish Sukinee /joint_states from real motor feedback. Type17 read-only."
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument(
        "--offset-json",
        default="/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json",
        help="Path to motor_pos -> URDF q software offset JSON.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Publish rate in Hz. Keep low for first verification.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print q_urdf every N publish cycles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.rate <= 0.0 or args.rate > 5.0:
        print("ERROR: --rate must be >0 and <=5 for this first read-only verification.")
        return 2

    rclpy.init()
    node = None

    try:
        node = SukineeRealFeedbackJointStatePublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopped by user. No real motor command was sent.")
    except Exception as e:
        print(f"RESULT: FAIL: {e}")
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())