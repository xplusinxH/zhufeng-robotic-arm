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


CAN_IFACE_DEFAULT = "can0"
HOST_ID = 0xFD
COMM_READ_PARAM = 17

READ_TIMEOUT_SEC = 2.0
MAX_READ_ATTEMPTS = 2
RETRY_DELAY_SEC = 0.10
INTER_PARAM_DELAY_SEC = 0.04
INTER_MOTOR_DELAY_SEC = 0.02

ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

PARAMS = [
    (0x7019, "pos", "rad"),
    (0x701A, "iqf", "A"),
    (0x701B, "vel", "rad/s"),
    (0x701C, "vbus", "V"),
]

# Verified by real feedback -> RViz direction check in Gate5.
# q_urdf = sign * (motor_pos - software_zero)
MOTOR_TO_URDF_SIGN = {
    1: 1.0,
    2: -1.0,
    3: 1.0,
    4:  1.0,
    5: 1.0,
    6:  1.0,
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


def read_all_feedback(can_iface: str) -> Tuple[Dict[int, Dict[str, Optional[float]]], Dict[int, Dict[str, str]]]:
    values: Dict[int, Dict[str, Optional[float]]] = {}
    statuses: Dict[int, Dict[str, str]] = {}

    reader = CandumpReader(can_iface)

    try:
        for motor_id in ALL_MOTOR_IDS:
            values[motor_id] = {}
            statuses[motor_id] = {}

            print(f"Joint{motor_id} CAN_ID={motor_id}")
            for index, name, unit in PARAMS:
                status, value = read_param(reader, can_iface, motor_id, index)
                statuses[motor_id][name] = status
                values[motor_id][name] = value

                if status == "OK":
                    print(f"  {name:<4}: {value: .6f} {unit}")
                else:
                    print(f"  {name:<4}: {status}")

                time.sleep(INTER_PARAM_DELAY_SEC)

            time.sleep(INTER_MOTOR_DELAY_SEC)
    finally:
        reader.close()

    return values, statuses


def check_feedback(values, statuses) -> bool:
    ok = True

    print()
    print("Read-only feedback sanity check:")

    for motor_id in ALL_MOTOR_IDS:
        for _, name, _ in PARAMS:
            if statuses.get(motor_id, {}).get(name) != "OK":
                print(f"  Joint{motor_id} {name}: FAIL, status={statuses[motor_id].get(name)}")
                ok = False

        if statuses.get(motor_id, {}).get("pos") != "OK":
            continue

        vbus = values[motor_id].get("vbus")
        iqf = values[motor_id].get("iqf")
        vel = values[motor_id].get("vel")

        if vbus is None or iqf is None or vel is None:
            continue

        vbus_ok = 10.0 <= float(vbus) <= 60.0
        iqf_ok = abs(float(iqf)) <= 1.0
        vel_ok = abs(float(vel)) <= 1.0

        print(
            f"  Joint{motor_id}: "
            f"vbus={'OK' if vbus_ok else 'WARN'}({float(vbus):.3f} V), "
            f"iqf={'OK' if iqf_ok else 'WARN'}({float(iqf):.3f} A), "
            f"vel={'OK' if vel_ok else 'WARN'}({float(vel):.3f} rad/s)"
        )

    print()
    print("FEEDBACK CHECK:", "PASS" if ok else "FAIL")
    return ok


def parse_six_floats(text: str, name: str):
    try:
        vals = [float(x.strip()) for x in text.split(",")]
    except Exception as e:
        raise ValueError(f"failed to parse {name}: {e}")

    if len(vals) != 6:
        raise ValueError(f"{name} must contain exactly 6 comma-separated values")

    return {i + 1: vals[i] for i in range(6)}


def compute_offsets(motor_pos: Dict[int, float], q_init: Dict[int, float]) -> Dict[int, float]:
    offsets = {}

    for motor_id in ARM_JOINT_IDS:
        sign = MOTOR_TO_URDF_SIGN[motor_id]
        offsets[motor_id] = motor_pos[motor_id] - sign * q_init[motor_id]

    return offsets


def compute_q_check(motor_pos: Dict[int, float], offsets: Dict[int, float]) -> Dict[int, float]:
    q_check = {}

    for motor_id in ARM_JOINT_IDS:
        sign = MOTOR_TO_URDF_SIGN[motor_id]
        q_check[motor_id] = sign * (motor_pos[motor_id] - offsets[motor_id])

    return q_check


def format_python_dict(name: str, data: Dict[int, float]) -> str:
    lines = [f"{name} = {{"]
    for motor_id in ARM_JOINT_IDS:
        lines.append(f"    {motor_id}: {data[motor_id]: .9f},")
    lines.append("}")
    return "\n".join(lines)


def save_json_config(path: Path, motor_pos: Dict[int, float], q_init: Dict[int, float], offsets: Dict[int, float]):
    payload = {
        "description": (
            "Software-only motor feedback position to URDF joint angle calibration. "
            "This does not set motor zero and does not change motor parameters."
        ),
        "formula": "q_urdf[j] = sign[j] * (motor_pos[j] - motor_pos_at_urdf_zero[j])",
        "joint_order": ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"],
        "motor_to_urdf_sign": {f"Joint{k}": MOTOR_TO_URDF_SIGN[k] for k in ARM_JOINT_IDS},
        "calibrated_motor_pos": {f"Joint{k}": motor_pos[k] for k in ARM_JOINT_IDS},
        "q_init_urdf": {f"Joint{k}": q_init[k] for k in ARM_JOINT_IDS},
        "motor_pos_at_urdf_zero": {f"Joint{k}": offsets[k] for k in ARM_JOINT_IDS},
        "created_at_unix_time": time.time(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate software offsets from RobStride motor feedback pos to URDF q. "
            "This script is read-only on CAN and does not set motor zero."
        )
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument(
        "--q-init",
        required=True,
        help=(
            "URDF q_init for Joint1-Joint6, comma-separated, in rad. "
            "Example: --q-init 0.1,1.2,-0.6,0.0,0.2,0.0"
        ),
    )
    parser.add_argument(
        "--motor-pos",
        default="",
        help=(
            "Optional manual motor feedback pos for Joint1-Joint6, comma-separated. "
            "If omitted, values are read from CAN using Type17."
        ),
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save calibration JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Sukinee motor-to-URDF software offset calibration")
    print()
    print("Safety status:")
    print("  NO Type1 motion command")
    print("  NO Type3 enable")
    print("  NO Type4 disable")
    print("  NO Type18 parameter write")
    print("  NO Type6 zero setting")
    print("  NO save parameters")
    print("  NO CAN_ID change")
    print("  NO protocol change")
    print("  NO MoveIt real execution")
    print("  CAN usage is Type17 read-only if --motor-pos is not provided.")
    print()

    try:
        q_init = parse_six_floats(args.q_init, "--q-init")
    except Exception as e:
        print("RESULT: FAIL")
        print(str(e))
        return 2

    print("Input q_init URDF pose:")
    for motor_id in ARM_JOINT_IDS:
        print(f"  Joint{motor_id}: {q_init[motor_id]:+.9f} rad")

    if args.motor_pos:
        try:
            motor_pos = parse_six_floats(args.motor_pos, "--motor-pos")
        except Exception as e:
            print("RESULT: FAIL")
            print(str(e))
            return 2

        print()
        print("Using manual motor feedback pos from --motor-pos.")
    else:
        print()
        print("=" * 90)
        print("Reading current Joint1-Joint7 feedback, Type17 read-only")
        print("=" * 90)

        values, statuses = read_all_feedback(args.can)
        if not check_feedback(values, statuses):
            print("RESULT: FAIL")
            print("Read-only feedback failed. Offsets were not computed.")
            return 1

        motor_pos = {
            motor_id: float(values[motor_id]["pos"])
            for motor_id in ARM_JOINT_IDS
        }

    print()
    print("Motor feedback pos used for calibration:")
    for motor_id in ARM_JOINT_IDS:
        print(f"  Joint{motor_id}: {motor_pos[motor_id]:+.9f} rad")

    offsets = compute_offsets(motor_pos, q_init)
    q_check = compute_q_check(motor_pos, offsets)

    print()
    print("=" * 90)
    print("Calibration result")
    print("=" * 90)
    print("Formula:")
    print("  q_urdf[j] = sign[j] * (motor_pos[j] - motor_pos_at_urdf_zero[j])")
    print()

    print("MOTOR_TO_URDF_SIGN:")
    for motor_id in ARM_JOINT_IDS:
        print(f"  Joint{motor_id}: {MOTOR_TO_URDF_SIGN[motor_id]:+.1f}")

    print()
    print("Computed MOTOR_POS_AT_URDF_ZERO:")
    for motor_id in ARM_JOINT_IDS:
        print(f"  Joint{motor_id}: {offsets[motor_id]:+.9f}")

    print()
    print("q_check from current motor_pos and computed offsets:")
    max_err = 0.0
    for motor_id in ARM_JOINT_IDS:
        err = q_check[motor_id] - q_init[motor_id]
        max_err = max(max_err, abs(err))
        print(
            f"  Joint{motor_id}: "
            f"q_check={q_check[motor_id]:+.9f}, "
            f"q_init={q_init[motor_id]:+.9f}, "
            f"err={err:+.9e}"
        )

    print()
    print("Copyable Python constants:")
    print()
    print(format_python_dict("MOTOR_TO_URDF_SIGN", MOTOR_TO_URDF_SIGN))
    print()
    print(format_python_dict("MOTOR_POS_AT_URDF_ZERO", offsets))

    if args.save_json:
        path = Path(args.save_json)
        save_json_config(path, motor_pos, q_init, offsets)
        print()
        print(f"Saved calibration JSON: {path}")

    print()
    if max_err < 1e-8:
        print("RESULT: PASS")
    else:
        print("RESULT: WARN")
        print("q_check does not exactly match q_init; inspect values above.")

    print("No real motor command was sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

