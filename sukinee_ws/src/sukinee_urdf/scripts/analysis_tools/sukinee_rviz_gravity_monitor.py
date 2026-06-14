#!/usr/bin/env python3
"""
RViz / joint_state_publisher_gui gravity monitor for Sukinee.

Purpose:
  Subscribe to /joint_states from joint_state_publisher_gui,
  use the GUI joint angles as URDF q,
  compute Pinocchio gravity torque in real time,
  and print raw_tau / corrected_tau / motor_ff.

Safety:
  - NO CAN access.
  - NO Type1 command.
  - NO Type3 enable.
  - NO Type4 disable.
  - NO Type17 read.
  - NO Type18 write.
  - NO motor zero setting.
  - NO real motor command.
  - This script only subscribes /joint_states and computes gravity torque.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_URDF = "/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf"
DEFAULT_CONFIG_JSON = "/home/zzj/sukinee_ws/sukinee_gravity_assist_config.json"
DEFAULT_INERTIA_CORRECTION_JSON = "/home/zzj/sukinee_ws/sukinee_inertia_correction.json"
DEFAULT_OFFSET_JSON = "/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json"

ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]
ARM_JOINT_NAMES = [f"Joint{i}" for i in ARM_JOINT_IDS]


def parse_joint_key_dict(raw: Dict[str, float]) -> Dict[int, float]:
    parsed: Dict[int, float] = {}

    for key, value in raw.items():
        if isinstance(key, str) and key.startswith("Joint"):
            motor_id = int(key.replace("Joint", ""))
        else:
            motor_id = int(key)

        parsed[motor_id] = float(value)

    return parsed


def load_offset_sign_as_torque_sign(path: Path) -> Dict[int, float]:
    """
    For this monitor only:
      use motor_to_urdf_sign as the default torque output sign preview.

    After torque_output_sign is formally added, this should be replaced by
    a dedicated torque_output_sign JSON field.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "motor_to_urdf_sign" not in payload:
        raise RuntimeError("offset JSON missing key: motor_to_urdf_sign")

    sign = parse_joint_key_dict(payload["motor_to_urdf_sign"])

    for mid in ARM_JOINT_IDS:
        if mid not in sign:
            raise RuntimeError(f"offset JSON missing sign for Joint{mid}")

    return sign


def load_gravity_config(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    gravity_feedforward_ratio = float(payload.get("gravity_feedforward_ratio", 1.0))
    gravity_joint_scale = parse_joint_key_dict(payload.get("gravity_joint_scale", {}))

    for mid in ARM_JOINT_IDS:
        gravity_joint_scale.setdefault(mid, 1.0)

    return {
        "gravity_feedforward_ratio": gravity_feedforward_ratio,
        "gravity_joint_scale": gravity_joint_scale,
        "raw": payload,
    }


def load_inertia_correction(path: Path):
    if not path.exists():
        return {
            "enabled": False,
            "joint_body_mass_scale": {},
            "raw": {},
            "message": f"inertia correction JSON not found: {path}",
        }

    payload = json.loads(path.read_text(encoding="utf-8"))

    enabled = bool(payload.get("enabled", False))
    joint_body_mass_scale = parse_joint_key_dict(payload.get("joint_body_mass_scale", {}))

    return {
        "enabled": enabled,
        "joint_body_mass_scale": joint_body_mass_scale,
        "raw": payload,
        "message": "loaded",
    }


def apply_inertia_correction_to_model(model: pin.Model, correction):
    """
    Return a model copy with selected joint body inertias mass-scaled.
    This does NOT modify URDF files.
    """
    corrected = pin.Model(model)

    applied = []
    if not correction.get("enabled", False):
        return corrected, applied

    joint_body_mass_scale = correction.get("joint_body_mass_scale", {})

    for mid, scale in joint_body_mass_scale.items():
        mid = int(mid)
        scale = float(scale)

        if mid == 1:
            applied.append(f"Joint1 skipped: base yaw")
            continue
        if mid == 7:
            applied.append(f"Joint7 skipped: gripper drive")
            continue
        if mid not in ARM_JOINT_IDS:
            applied.append(f"Joint{mid} skipped: not in arm joints")
            continue

        joint_name = f"Joint{mid}"
        if not corrected.existJointName(joint_name):
            applied.append(f"{joint_name}: not found")
            continue

        if scale <= 0.0 or scale > 3.0:
            raise RuntimeError(f"{joint_name} mass scale must be >0 and <=3.0, got {scale}")

        jid = corrected.getJointId(joint_name)
        inertia = corrected.inertias[jid]

        old_mass = float(inertia.mass)
        if abs(scale - 1.0) < 1e-12:
            applied.append(f"{joint_name}: scale=1.000; no change")
            continue

        new_mass = old_mass * scale
        new_lever = np.array(inertia.lever, dtype=float)
        new_rot_inertia = np.array(inertia.inertia, dtype=float) * scale

        corrected.inertias[jid] = pin.Inertia(new_mass, new_lever, new_rot_inertia)

        applied.append(f"{joint_name}: scale={scale:.3f}, mass {old_mass:.6f} -> {new_mass:.6f} kg")

    return corrected, applied


def total_model_mass(model: pin.Model) -> float:
    return float(sum(float(model.inertias[jid].mass) for jid in range(1, model.njoints)))


def load_pinocchio_models(urdf_path: Path, inertia_correction):
    if not urdf_path.exists():
        raise RuntimeError(f"URDF not found: {urdf_path}")

    raw_model = pin.buildModelFromUrdf(str(urdf_path))
    raw_data = raw_model.createData()

    corrected_model, applied = apply_inertia_correction_to_model(raw_model, inertia_correction)
    corrected_data = corrected_model.createData()

    joint_index = {}
    for mid in ARM_JOINT_IDS:
        joint_name = f"Joint{mid}"
        if not raw_model.existJointName(joint_name):
            raise RuntimeError(f"Pinocchio joint not found: {joint_name}")

        jid = raw_model.getJointId(joint_name)
        joint_index[mid] = {
            "jid": jid,
            "idx_q": int(raw_model.idx_qs[jid]),
            "idx_v": int(raw_model.idx_vs[jid]),
            "nq": int(raw_model.nqs[jid]),
            "nv": int(raw_model.nvs[jid]),
        }

    return raw_model, raw_data, corrected_model, corrected_data, joint_index, applied


def write_joint_to_pin_q(model: pin.Model, q: np.ndarray, joint_index, mid: int, angle: float):
    info = joint_index[mid]
    idx_q = info["idx_q"]
    nq = info["nq"]

    if nq == 1:
        q[idx_q] = float(angle)
    elif nq == 2:
        # Pinocchio continuous revolute joint uses cos/sin representation.
        q[idx_q] = math.cos(float(angle))
        q[idx_q + 1] = math.sin(float(angle))
    else:
        raise RuntimeError(f"Unsupported Joint{mid} nq={nq}")


def compute_gravity_tau(model, data, joint_index, q_urdf: Dict[int, float]):
    q = pin.neutral(model)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)

    for mid in ARM_JOINT_IDS:
        write_joint_to_pin_q(model, q, joint_index, mid, q_urdf.get(mid, 0.0))

    tau_vec = pin.rnea(model, data, q, v, a)

    tau = {}
    for mid in ARM_JOINT_IDS:
        idx_v = joint_index[mid]["idx_v"]
        tau[mid] = float(tau_vec[idx_v])

    return tau


def format_deg(rad: float) -> str:
    return f"{math.degrees(rad):+7.2f}°"


class RvizGravityMonitor(Node):
    def __init__(self, args):
        super().__init__("sukinee_rviz_gravity_monitor")

        self.urdf_path = Path(args.urdf).expanduser()
        self.config_path = Path(args.config_json).expanduser()
        self.inertia_correction_path = Path(args.inertia_correction_json).expanduser()
        self.offset_path = Path(args.offset_json).expanduser()

        self.print_every_sec = float(args.print_every_sec)
        self.last_print_time = 0.0

        self.gravity_config = load_gravity_config(self.config_path)
        self.torque_sign = load_offset_sign_as_torque_sign(self.offset_path)
        self.inertia_correction = load_inertia_correction(self.inertia_correction_path)

        (
            self.raw_model,
            self.raw_data,
            self.corrected_model,
            self.corrected_data,
            self.joint_index,
            self.correction_applied,
        ) = load_pinocchio_models(self.urdf_path, self.inertia_correction)

        self.raw_mass = total_model_mass(self.raw_model)
        self.corrected_mass = total_model_mass(self.corrected_model)

        self.get_logger().info("Sukinee RViz gravity monitor started.")
        self.print_startup_summary()

        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_state,
            10,
        )

    def print_startup_summary(self):
        print()
        print("=" * 100)
        print("Sukinee RViz / joint_state_publisher_gui gravity monitor")
        print("=" * 100)
        print("Safety status:")
        print("  NO CAN access")
        print("  NO Type1 motion command")
        print("  NO Type3 enable")
        print("  NO Type4 disable")
        print("  NO Type17 read")
        print("  NO Type18 write")
        print("  NO motor zero setting")
        print("  This script only subscribes /joint_states and computes gravity torque.")
        print()
        print("Loaded files:")
        print(f"  URDF:                    {self.urdf_path}")
        print(f"  config JSON:             {self.config_path}")
        print(f"  offset JSON:             {self.offset_path}")
        print(f"  inertia correction JSON: {self.inertia_correction_path}")
        print()
        print("Model mass:")
        print(f"  raw_total_mass:       {self.raw_mass:.6f} kg")
        print(f"  corrected_total_mass: {self.corrected_mass:.6f} kg")
        print(f"  delta_total_mass:     {self.corrected_mass - self.raw_mass:+.6f} kg")
        print()
        print("Inertia correction:")
        print(f"  enabled: {self.inertia_correction.get('enabled', False)}")
        for item in self.correction_applied:
            print(f"  {item}")
        print()
        print("Gravity config:")
        print(f"  gravity_feedforward_ratio = {self.gravity_config['gravity_feedforward_ratio']:.3f}")
        for mid in ARM_JOINT_IDS:
            print(
                f"  Joint{mid}: gravity_joint_scale={self.gravity_config['gravity_joint_scale'][mid]:.3f}, "
                f"torque_sign_preview={self.torque_sign[mid]:+.1f}"
            )
        print()
        print("Move the sliders in joint_state_publisher_gui.")
        print("Watch corrected_tau and motor_ff_preview signs/magnitudes.")
        print("=" * 100)
        print()

    def on_joint_state(self, msg: JointState):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_print_time < self.print_every_sec:
            return
        self.last_print_time = now

        name_to_pos = dict(zip(msg.name, msg.position))

        q_urdf = {}
        missing = []
        for mid in ARM_JOINT_IDS:
            name = f"Joint{mid}"
            if name not in name_to_pos:
                missing.append(name)
                q_urdf[mid] = 0.0
            else:
                q_urdf[mid] = float(name_to_pos[name])

        if missing:
            print(f"WARNING: /joint_states missing joints: {missing}")

        raw_tau = compute_gravity_tau(
            self.raw_model,
            self.raw_data,
            self.joint_index,
            q_urdf,
        )
        corrected_tau = compute_gravity_tau(
            self.corrected_model,
            self.corrected_data,
            self.joint_index,
            q_urdf,
        )

        gravity_feedforward_ratio = self.gravity_config["gravity_feedforward_ratio"]
        gravity_joint_scale = self.gravity_config["gravity_joint_scale"]

        motor_ff = {}
        for mid in ARM_JOINT_IDS:
            motor_ff[mid] = (
                float(self.torque_sign[mid])
                * float(corrected_tau[mid])
                * float(gravity_feedforward_ratio)
                * float(gravity_joint_scale[mid])
            )

        print()
        print("-" * 130)
        print("RViz GUI gravity preview from /joint_states")
        print("-" * 130)
        print(
            "Joint | q(rad)     | q(deg)    | raw_tau(Nm) | corrected_tau(Nm) | delta(Nm) | motor_ff_preview(Nm)"
        )
        print("-" * 130)

        for mid in ARM_JOINT_IDS:
            delta = corrected_tau[mid] - raw_tau[mid]
            print(
                f"J{mid:<4} | "
                f"{q_urdf[mid]:+10.6f} | "
                f"{format_deg(q_urdf[mid]):>9} | "
                f"{raw_tau[mid]:+11.6f} | "
                f"{corrected_tau[mid]:+17.6f} | "
                f"{delta:+9.6f} | "
                f"{motor_ff[mid]:+20.6f}"
            )

        print("-" * 130)
        print(
            "Focus: move Joint2 slowly from 0° toward +90°. "
            "Check whether J2 corrected_tau / motor_ff crosses zero or changes direction."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor Pinocchio gravity torque from RViz joint_state_publisher_gui /joint_states."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--config-json", default=DEFAULT_CONFIG_JSON)
    parser.add_argument("--inertia-correction-json", default=DEFAULT_INERTIA_CORRECTION_JSON)
    parser.add_argument("--offset-json", default=DEFAULT_OFFSET_JSON)
    parser.add_argument("--print-every-sec", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rclpy.init()
    node = RvizGravityMonitor(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())