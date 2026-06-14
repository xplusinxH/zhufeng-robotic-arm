#!/usr/bin/env python3
import argparse
import math
import json
import re
import select
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CAN_IFACE_DEFAULT = "can0"
DEFAULT_INERTIA_CORRECTION_JSON = "/home/zzj/sukinee_ws/sukinee_inertia_correction.json"
HOST_ID = 0xFD
COMM_READ_PARAM = 17

READ_TIMEOUT_SEC = 2.0
MAX_READ_ATTEMPTS = 2
RETRY_DELAY_SEC = 0.10
INTER_PARAM_DELAY_SEC = 0.04
INTER_MOTOR_DELAY_SEC = 0.02

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]
ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]

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

# Fallback motor-feedback-pos -> URDF-q mapping.
# These constants are only used if --offset-json is not provided.
#
# IMPORTANT:
#   q_urdf = MOTOR_TO_URDF_SIGN[j] * (motor_pos - MOTOR_POS_AT_URDF_ZERO[j])
#
# For the current Gate5 workflow, prefer loading:
#   /home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json
#
# This is software mapping only. It does NOT set motor zero.
MOTOR_TO_URDF_SIGN = {
    1:  1.0,
    2: -1.0,
    3:  1.0,
    4:  1.0,
    5:  1.0,
    6:  1.0,
}

# Software offset only. Do NOT confuse this with motor zero setting.
# Keep as 0.0 for first preview. Later, replace with measured motor feedback
# values at a known URDF pose.
MOTOR_POS_AT_URDF_ZERO = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
}

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


def build_extended_can_id(comm_type: int, data_area2: int, target_id: int) -> int:
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
        for motor_id in MOTOR_IDS:
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

    for motor_id in MOTOR_IDS:
        for _, name, _ in PARAMS:
            if statuses.get(motor_id, {}).get(name) != "OK":
                print(f"  Joint{motor_id} {name}: FAIL, status={statuses[motor_id].get(name)}")
                ok = False

        if not ok:
            continue

        vbus = float(values[motor_id]["vbus"])
        iqf = float(values[motor_id]["iqf"])
        vel = float(values[motor_id]["vel"])

        vbus_ok = 10.0 <= vbus <= 60.0
        iqf_ok = abs(iqf) <= 1.0
        vel_ok = abs(vel) <= 1.0

        print(
            f"  Joint{motor_id}: "
            f"vbus={'OK' if vbus_ok else 'WARN'}({vbus:.3f} V), "
            f"iqf={'OK' if iqf_ok else 'WARN'}({iqf:.3f} A), "
            f"vel={'OK' if vel_ok else 'WARN'}({vel:.3f} rad/s)"
        )

    print()
    print("FEEDBACK CHECK:", "PASS" if ok else "FAIL")
    return ok


def find_default_urdf() -> Optional[Path]:
    candidates = [
        Path("/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf"),
        Path("/home/zzj/sukinee_ws/install/sukinee_urdf/share/sukinee_urdf/urdf/sukinee_urdf.urdf"),
        Path("/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf.xacro"),
        Path("/home/zzj/sukinee_ws/install/sukinee_urdf/share/sukinee_urdf/urdf/sukinee_urdf.urdf.xacro"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def parse_q_arg(q_text: str):
    try:
        vals = [float(x.strip()) for x in q_text.split(",")]
    except Exception as e:
        raise ValueError(f"failed to parse --q: {e}")

    if len(vals) != 6:
        raise ValueError("--q must contain exactly 6 comma-separated values")

    return {i + 1: vals[i] for i in range(6)}


def map_motor_feedback_to_urdf_q(values) -> Dict[int, float]:
    q_map: Dict[int, float] = {}

    for motor_id in ARM_JOINT_IDS:
        motor_pos = float(values[motor_id]["pos"])
        sign = MOTOR_TO_URDF_SIGN[motor_id]
        zero = MOTOR_POS_AT_URDF_ZERO[motor_id]
        q_map[motor_id] = sign * (motor_pos - zero)

    return q_map


def load_pinocchio_model(urdf_path: Path):
    try:
        import pinocchio as pin
        import numpy as np
    except Exception as e:
        print("ERROR: failed to import pinocchio or numpy.")
        print("Try checking:")
        print('  python3 -c "import pinocchio as pin; print(pin.__version__)"')
        raise e

    if not urdf_path.exists():
        raise FileNotFoundError(str(urdf_path))

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return pin, np, model, data


def build_pinocchio_q(pin, np, model, q_map: Dict[int, float]):
    # Use Pinocchio neutral configuration.
    # This is important for continuous joints, whose q representation is [cos(theta), sin(theta)].
    q = pin.neutral(model)

    for motor_id, joint_name in JOINT_NAME_BY_ID.items():
        if not model.existJointName(joint_name):
            raise RuntimeError(
                f"URDF / Pinocchio model does not contain joint name: {joint_name}"
            )

        joint_id = model.getJointId(joint_name)
        nq = model.nqs[joint_id]
        nv = model.nvs[joint_id]
        idx_q = model.idx_qs[joint_id]

        theta = q_map[motor_id]

        if nq == 1 and nv == 1:
            # Normal revolute joint.
            q[idx_q] = theta
        elif nq == 2 and nv == 1:
            # Continuous revolute joint in Pinocchio.
            # Configuration is represented as [cos(theta), sin(theta)].
            q[idx_q] = math.cos(theta)
            q[idx_q + 1] = math.sin(theta)
        else:
            raise RuntimeError(
                f"{joint_name} has nq={nq}, nv={nv}; "
                "this preview script supports only revolute nq=1,nv=1 "
                "or continuous revolute nq=2,nv=1 joints."
            )

    return q




def model_total_mass(model) -> float:
    total = 0.0
    for jid in range(1, model.njoints):
        total += float(model.inertias[jid].mass)
    return total


def load_inertia_correction_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path:
        return None

    if not path.exists():
        raise RuntimeError(f"inertia correction JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    enabled = bool(payload.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "path": str(path),
            "raw": payload,
            "joint_body_mass_scale": {},
            "extra_payload": {"enabled": False},
        }

    safety = payload.get("safety", {})
    max_allowed_mass_scale = float(safety.get("max_allowed_mass_scale", 3.0))
    if max_allowed_mass_scale <= 0.0 or max_allowed_mass_scale > 10.0:
        raise RuntimeError("safety.max_allowed_mass_scale must be >0 and <=10.0")

    raw_scales = payload.get("joint_body_mass_scale", {})
    if not isinstance(raw_scales, dict):
        raise RuntimeError("joint_body_mass_scale must be a dictionary")

    joint_body_mass_scale: Dict[str, float] = {}
    for joint_name, scale_raw in raw_scales.items():
        joint_name = str(joint_name)
        scale = float(scale_raw)

        if joint_name in ("Joint1", "Joint7", "left_finger", "right_finger"):
            raise RuntimeError(f"{joint_name} must not be mass-scaled in this stage")
        if joint_name not in ["Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]:
            raise RuntimeError(
                f"unsupported correction joint key: {joint_name}; "
                "supported now: Joint2-Joint6"
            )
        if scale <= 0.0 or scale > max_allowed_mass_scale:
            raise RuntimeError(
                f"{joint_name} mass scale {scale} out of range; "
                f"allowed: >0 and <= {max_allowed_mass_scale}"
            )

        joint_body_mass_scale[joint_name] = scale

    extra_payload = payload.get("extra_payload", {"enabled": False})
    if not isinstance(extra_payload, dict):
        raise RuntimeError("extra_payload must be a dictionary")

    if bool(extra_payload.get("enabled", False)):
        parent_joint = str(extra_payload.get("parent_joint", ""))
        if parent_joint not in ["Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]:
            raise RuntimeError("extra_payload.parent_joint must be one of Joint2-Joint6")

        mass_kg = float(extra_payload.get("mass_kg", 0.0))
        if mass_kg <= 0.0 or mass_kg > 2.0:
            raise RuntimeError("extra_payload.mass_kg must be >0 and <=2.0 for this stage")

        local_com = extra_payload.get("local_com_xyz_m", None)
        if not isinstance(local_com, list) or len(local_com) != 3:
            raise RuntimeError("extra_payload.local_com_xyz_m must be a list of three numbers")
        extra_payload["local_com_xyz_m"] = [float(x) for x in local_com]
        extra_payload["mass_kg"] = mass_kg
        extra_payload["parent_joint"] = parent_joint

    return {
        "enabled": True,
        "path": str(path),
        "raw": payload,
        "joint_body_mass_scale": joint_body_mass_scale,
        "extra_payload": extra_payload,
        "max_allowed_mass_scale": max_allowed_mass_scale,
    }


def make_corrected_pinocchio_model(pin, np, model, correction: Optional[Dict[str, Any]]):
    """
    Return a corrected copy of the Pinocchio model.

    Stage-1 correction strategy:
      - scale selected Pinocchio joint-body inertias Joint2-Joint6;
      - keep COM lever unchanged;
      - scale rotational inertia linearly with mass;
      - optional point-mass payload can be added to a selected parent joint frame.

    This does not modify the URDF file and does not command motors.
    """
    try:
        corrected_model = model.copy()
    except Exception:
        import copy
        corrected_model = copy.deepcopy(model)

    applied: List[Dict[str, Any]] = []

    if not correction or not correction.get("enabled", False):
        return corrected_model, applied

    for joint_name, scale in correction.get("joint_body_mass_scale", {}).items():
        if abs(scale - 1.0) < 1e-12:
            applied.append({
                "type": "mass_scale",
                "joint_name": joint_name,
                "joint_id": int(model.getJointId(joint_name)) if model.existJointName(joint_name) else -1,
                "scale": scale,
                "old_mass_kg": None,
                "new_mass_kg": None,
                "note": "scale is 1.0; no physical change",
            })
            continue

        if not corrected_model.existJointName(joint_name):
            raise RuntimeError(f"Pinocchio model does not contain joint: {joint_name}")

        jid = corrected_model.getJointId(joint_name)
        old_inertia = corrected_model.inertias[jid]
        old_mass = float(old_inertia.mass)
        new_mass = old_mass * float(scale)

        lever = np.array(old_inertia.lever, dtype=float).copy()
        rotational = np.array(old_inertia.inertia, dtype=float).copy() * float(scale)
        corrected_model.inertias[jid] = pin.Inertia(new_mass, lever, rotational)

        applied.append({
            "type": "mass_scale",
            "joint_name": joint_name,
            "joint_id": int(jid),
            "scale": float(scale),
            "old_mass_kg": old_mass,
            "new_mass_kg": new_mass,
            "note": "scaled body mass and rotational inertia; COM lever unchanged",
        })

    extra_payload = correction.get("extra_payload", {"enabled": False})
    if bool(extra_payload.get("enabled", False)):
        parent_joint = str(extra_payload["parent_joint"])
        if not corrected_model.existJointName(parent_joint):
            raise RuntimeError(f"Pinocchio model does not contain payload parent joint: {parent_joint}")

        jid = corrected_model.getJointId(parent_joint)
        mass_kg = float(extra_payload["mass_kg"])
        local_com = np.array(extra_payload["local_com_xyz_m"], dtype=float)
        zero_rot = np.zeros((3, 3))
        payload_inertia = pin.Inertia(mass_kg, local_com, zero_rot)
        corrected_model.inertias[jid] = corrected_model.inertias[jid] + payload_inertia

        applied.append({
            "type": "extra_payload_point_mass",
            "joint_name": parent_joint,
            "joint_id": int(jid),
            "mass_kg": mass_kg,
            "local_com_xyz_m": [float(x) for x in local_com],
            "note": "added point-mass payload to selected joint body for preview",
        })

    return corrected_model, applied


def print_inertia_correction_summary(correction: Optional[Dict[str, Any]], applied: List[Dict[str, Any]], raw_mass: float, corrected_mass: float):
    print()
    print("=" * 90)
    print("External inertia correction summary")
    print("=" * 90)

    if not correction:
        print("No inertia correction JSON was provided. Using raw URDF inertials only.")
        return

    print(f"correction JSON: {correction.get('path', '')}")
    print(f"enabled: {bool(correction.get('enabled', False))}")
    print(f"raw total mass:       {raw_mass:.6f} kg")
    print(f"corrected total mass: {corrected_mass:.6f} kg")
    print(f"delta total mass:     {corrected_mass - raw_mass:+.6f} kg")

    if not correction.get("enabled", False):
        print("Correction file exists but enabled=false, so raw URDF inertials are used.")
        return

    if not applied:
        print("Correction enabled, but no correction item was applied.")
        return

    print()
    print("Applied correction items:")
    for item in applied:
        if item["type"] == "mass_scale":
            old_mass = item.get("old_mass_kg")
            new_mass = item.get("new_mass_kg")
            if old_mass is None:
                print(
                    f"  {item['joint_name']}: mass scale={item['scale']:.3f}; "
                    f"{item['note']}"
                )
            else:
                print(
                    f"  {item['joint_name']}: mass scale={item['scale']:.3f}, "
                    f"mass {old_mass:.6f} -> {new_mass:.6f} kg"
                )
        elif item["type"] == "extra_payload_point_mass":
            print(
                f"  payload on {item['joint_name']}: "
                f"mass={item['mass_kg']:.6f} kg, "
                f"local_com={item['local_com_xyz_m']}"
            )

def compute_gravity_preview(urdf_path: Path, q_map: Dict[int, float], scales, correction: Optional[Dict[str, Any]] = None):
    pin, np, model_raw, data_raw = load_pinocchio_model(urdf_path)
    q_raw = build_pinocchio_q(pin, np, model_raw, q_map)

    model_corrected, applied_corrections = make_corrected_pinocchio_model(pin, np, model_raw, correction)
    data_corrected = model_corrected.createData()
    q_corrected = build_pinocchio_q(pin, np, model_corrected, q_map)

    zero_v_raw = np.zeros(model_raw.nv)
    zero_a_raw = np.zeros(model_raw.nv)
    zero_v_corrected = np.zeros(model_corrected.nv)
    zero_a_corrected = np.zeros(model_corrected.nv)

    tau_raw = pin.rnea(model_raw, data_raw, q_raw, zero_v_raw, zero_a_raw)
    tau_corrected = pin.rnea(
        model_corrected,
        data_corrected,
        q_corrected,
        zero_v_corrected,
        zero_a_corrected,
    )

    raw_total_mass = model_total_mass(model_raw)
    corrected_total_mass = model_total_mass(model_corrected)

    print()
    print("=" * 90)
    print("Pinocchio model summary")
    print("=" * 90)
    print(f"URDF path: {urdf_path}")
    print(f"model nq={model_raw.nq}, nv={model_raw.nv}, njoints={model_raw.njoints}")
    print(f"raw sum of body inertial masses: {raw_total_mass:.6f} kg")
    print(f"corrected sum of body inertial masses: {corrected_total_mass:.6f} kg")

    if raw_total_mass < 0.1:
        print("WARNING: total inertial mass is very small. URDF inertial parameters may be missing.")

    print_inertia_correction_summary(
        correction=correction,
        applied=applied_corrections,
        raw_mass=raw_total_mass,
        corrected_mass=corrected_total_mass,
    )

    print()
    print("URDF q used for gravity preview:")
    for motor_id in ARM_JOINT_IDS:
        print(
            f"  Joint{motor_id}: q_urdf={q_map[motor_id]:+.6f} rad "
            f"(sign={MOTOR_TO_URDF_SIGN[motor_id]:+.1f}, "
            f"software_zero={MOTOR_POS_AT_URDF_ZERO[motor_id]:+.6f})"
        )

    print()
    print("=" * 90)
    print("Gravity torque preview: raw URDF vs corrected model")
    print("=" * 90)
    print("Sign convention:")
    print("  tau_urdf is Pinocchio generalized torque in URDF joint coordinates.")
    print("  tau_motor_preview = MOTOR_TO_URDF_SIGN[j] * tau_corrected_urdf * scale.")
    print("  This is preview only. Do NOT send these torques before sign/zero/scale validation.")
    print()

    raw_tau_by_joint: Dict[int, float] = {}
    corrected_tau_by_joint: Dict[int, float] = {}

    for motor_id, joint_name in JOINT_NAME_BY_ID.items():
        joint_id_raw = model_raw.getJointId(joint_name)
        joint_id_corrected = model_corrected.getJointId(joint_name)
        idx_v_raw = model_raw.idx_vs[joint_id_raw]
        idx_v_corrected = model_corrected.idx_vs[joint_id_corrected]
        nv_raw = model_raw.nvs[joint_id_raw]
        nv_corrected = model_corrected.nvs[joint_id_corrected]

        if nv_raw != 1 or nv_corrected != 1:
            raise RuntimeError(f"{joint_name} has unexpected nv; expected 1.")

        raw_tau_by_joint[motor_id] = float(tau_raw[idx_v_raw])
        corrected_tau_by_joint[motor_id] = float(tau_corrected[idx_v_corrected])

    header = "Joint | raw_tau(Nm) | corrected_tau(Nm) | delta(Nm)"
    for scale in scales:
        header += f" | corrected_motor_ff@{scale:.2f}(Nm)"
    print(header)
    print("-" * len(header))

    for motor_id in ARM_JOINT_IDS:
        raw_tau = raw_tau_by_joint[motor_id]
        corrected_tau = corrected_tau_by_joint[motor_id]
        delta_tau = corrected_tau - raw_tau
        row = f"J{motor_id:<4} | {raw_tau:+.6f} | {corrected_tau:+.6f} | {delta_tau:+.6f}"
        for scale in scales:
            motor_ff = MOTOR_TO_URDF_SIGN[motor_id] * corrected_tau * scale
            row += f" | {motor_ff:+.6f}"
        print(row)

    print()
    max_abs_raw_tau = max(abs(raw_tau_by_joint[mid]) for mid in ARM_JOINT_IDS)
    max_abs_corrected_tau = max(abs(corrected_tau_by_joint[mid]) for mid in ARM_JOINT_IDS)
    print(f"Max abs raw tau_urdf:       {max_abs_raw_tau:.6f} Nm")
    print(f"Max abs corrected tau_urdf: {max_abs_corrected_tau:.6f} Nm")

    if max_abs_corrected_tau < 1e-6:
        print("WARNING: all corrected gravity torques are near zero.")
        print("Possible reasons: URDF inertials missing, q mapping wrong, gravity axis issue, or pose near zero-load.")

    print()
    print("NEXT SAFETY NOTE:")
    print("  This script did not send Type1 / Type3 / Type4 / Type18 control commands.")
    print("  Use this output only to inspect model/sign/scale before any zero-torque drag test.")

    return raw_tau_by_joint, corrected_tau_by_joint

def parse_scales(text: str):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one scale is required")
    for val in vals:
        if val < 0.0 or val > 1.0:
            raise ValueError("scales must be between 0.0 and 1.0")
    return vals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sukinee Pinocchio gravity preview. Read-only CAN + compute-only Pinocchio."
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument("--urdf", default="", help="Path to URDF. If omitted, common Sukinee paths are tried.")
    parser.add_argument("--no-can", action="store_true", help="Do not read CAN. Use --q or zeros.")
    parser.add_argument(
    "--offset-json",
    default="/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json",
    help="Path to motor_pos -> URDF q software offset JSON.",
)
    parser.add_argument(
        "--q",
        default="",
        help="Manual URDF q for Joint1-Joint6, comma-separated. Example: --q 0,0,0,0,0,0",
    )
    parser.add_argument(
        "--scales",
        default="0.1,0.2,0.3",
        help="Preview gravity feedforward scales, comma-separated. Default: 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--inertia-correction-json",
        default="",
        help=(
            "Optional external inertia correction JSON. "
            f"Example: {DEFAULT_INERTIA_CORRECTION_JSON}. "
            "If omitted, raw URDF inertials are used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global MOTOR_TO_URDF_SIGN
    global MOTOR_POS_AT_URDF_ZERO
    print("Sukinee gravity preview: READ-ONLY / COMPUTE-ONLY")
    print()
    print("Safety status:")
    print("  NO Type1 motion command")
    print("  NO Type3 enable")
    print("  NO Type4 disable")
    print("  NO Type18 parameter write")
    print("  NO zero setting")
    print("  NO save parameters")
    print("  NO MoveIt real execution")
    print("  CAN usage, if enabled, is Type17 read-only parameter reads.")
    print()

    scales = parse_scales(args.scales)
    if args.offset_json:
        offset_path = Path(args.offset_json)
        if not offset_path.exists():
            print("RESULT: FAIL")
            print(f"offset JSON not found: {offset_path}")
            return 2

        loaded_sign, loaded_zero, _payload = load_offset_json(offset_path)
        MOTOR_TO_URDF_SIGN = loaded_sign
        MOTOR_POS_AT_URDF_ZERO = loaded_zero

        print("Loaded motor-to-URDF software offset JSON:")
        print(f"  {offset_path}")
        for motor_id in ARM_JOINT_IDS:
            print(
                f"  Joint{motor_id}: "
                f"sign={MOTOR_TO_URDF_SIGN[motor_id]:+.1f}, "
                f"software_zero={MOTOR_POS_AT_URDF_ZERO[motor_id]:+.9f}"
            )
        print()

    if args.urdf:
        urdf_path = Path(args.urdf)
    else:
        found = find_default_urdf()
        if found is None:
            print("ERROR: could not find Sukinee URDF automatically.")
            print("Use --urdf /path/to/sukinee_urdf.urdf")
            return 2
        urdf_path = found

    if args.q:
        q_map = parse_q_arg(args.q)
        print("Using manual URDF q from --q.")
    elif args.no_can:
        q_map = {i: 0.0 for i in ARM_JOINT_IDS}
        print("Using zero URDF q because --no-can was provided and --q was not set.")
    else:
        print("=" * 90)
        print("Joint1-Joint7 read-only feedback")
        print("=" * 90)

        values, statuses = read_all_feedback(args.can)
        if not check_feedback(values, statuses):
            print("RESULT: FAIL")
            print("Read-only feedback failed. Do not use gravity preview.")
            return 1

        q_map = map_motor_feedback_to_urdf_q(values)

    print()
    print("Software mapping status:")
    print("  q_urdf = sign * (motor_feedback_pos - software_zero)")
    print("  sign / software_zero should come from the calibrated offset JSON.")
    print("  This script is still READ-ONLY / COMPUTE-ONLY. It does not send torque.")

    correction = None
    if args.inertia_correction_json:
        try:
            correction = load_inertia_correction_json(Path(args.inertia_correction_json))
        except Exception as e:
            print()
            print("RESULT: FAIL")
            print(f"Failed to load inertia correction JSON: {e}")
            return 2

    try:
        compute_gravity_preview(urdf_path, q_map, scales, correction=correction)
    except Exception as e:
        print()
        print("RESULT: FAIL")
        print(str(e))
        return 1

    print()
    print("RESULT: PASS")
    print("Gravity preview completed. No real motor command was sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())