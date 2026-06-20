#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pinocchio as pin

from sukinee_socketcan_driver import SukineeSocketCANDriver, CAN_IFACE_DEFAULT


ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

# Terminal monitor output is intentionally limited to Joint2-Joint4.
# This does NOT change the real control target_joints from config.
MONITOR_JOINT_IDS = [2, 3, 4]

# JointState names used only when --publish-joint-states is enabled.
# Keep this consistent with the current real feedback publisher.
JOINT_STATE_NAMES = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
    "Joint7",
]

POS_INDEX = 0x7019
PARAMS_ALL = [
    (0x7019, "pos"),
    (0x701A, "iqf"),
    (0x701B, "vel"),
    (0x701C, "vbus"),
]

ARMED_CONFIRM_TEXT = "I_UNDERSTAND_THIS_RUNS_SOCKETCAN_ZERO_TORQUE_GRAVITY_MODE"

DEFAULT_URDF = "/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf"
DEFAULT_OFFSET_JSON = "/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json"
DEFAULT_CONFIG_JSON = "/home/zzj/sukinee_ws/sukinee_gravity_assist_config.json"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * p / 100.0))
    idx = max(0, min(len(s) - 1, idx))
    return s[idx]


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


def load_gravity_config(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    target_joints = [int(x) for x in payload.get("target_joints", [2, 3, 4])]
    if not target_joints:
        raise RuntimeError("target_joints must not be empty")

    for motor_id in target_joints:
        if motor_id not in ARM_JOINT_IDS:
            raise RuntimeError(f"target_joints can only contain Joint1-Joint6, got Joint{motor_id}")
        if motor_id == 1:
            raise RuntimeError("Joint1 is not allowed in this first gravity mode controller.")
        if motor_id == 7:
            raise RuntimeError("Joint7 must not be used for arm gravity mode.")

    gravity_feedforward_ratio = float(payload.get("gravity_feedforward_ratio", 1.0))
    if gravity_feedforward_ratio <= 0.0 or gravity_feedforward_ratio > 2.0:
        raise RuntimeError("gravity_feedforward_ratio must be >0 and <=2.0 for this stage")

    gravity_joint_scale = parse_joint_key_dict(payload.get("gravity_joint_scale", {}))
    zero_torque_kd = parse_joint_key_dict(payload.get("zero_torque_kd", {}))
    max_abs_torque = parse_joint_key_dict(payload.get("max_abs_torque", {}))
    torque_slew_rate = parse_joint_key_dict(payload.get("torque_slew_rate", {}))

    software_damping_raw = dict(payload.get("software_damping", {}))
    software_damping_enabled = bool(software_damping_raw.get("enabled", False))
    software_damping_coeff = parse_joint_key_dict(
        software_damping_raw.get("damping", software_damping_raw.get("joint_damping", {}))
    )
    software_damping_max = parse_joint_key_dict(
        software_damping_raw.get("max_abs_damping_torque", {})
    )
    qdot_filter_alpha = float(software_damping_raw.get("qdot_filter_alpha", 0.25))
    max_abs_qdot = float(software_damping_raw.get("max_abs_qdot", 6.0))

    # Optional Joint2 anti-Joint3-coupling hold.
    # This is MOTOR-SIDE and is meant to prevent Joint2 from being dragged away
    # when Joint3 is producing a large gravity/feedforward torque.
    # It is a small gated virtual hold term, not a full position controller.
    joint2_anti_j3_hold_raw = dict(payload.get("joint2_anti_j3_coupling_hold", {}))
    joint2_anti_j3_hold = {
        "enabled": bool(joint2_anti_j3_hold_raw.get("enabled", False)),
        "q2_hold_mode": str(joint2_anti_j3_hold_raw.get("q2_hold_mode", "capture_at_start")),
        "j3_abs_torque_start": float(joint2_anti_j3_hold_raw.get("j3_abs_torque_start", 2.0)),
        "j3_abs_torque_full": float(joint2_anti_j3_hold_raw.get("j3_abs_torque_full", 4.0)),
        "kp": float(joint2_anti_j3_hold_raw.get("kp", 0.0)),
        "kd": float(joint2_anti_j3_hold_raw.get("kd", 0.0)),
        "max_abs_hold_torque": float(joint2_anti_j3_hold_raw.get("max_abs_hold_torque", 0.0)),
        "raw": joint2_anti_j3_hold_raw,
    }

    # Optional Joint4 angle-dependent gravity scale.
    # Purpose:
    #   Joint4 can be over-compensated at positive q4 angles. This block can
    #   reduce or increase only Joint4 gravity gain smoothly as q4 changes.
    #   It does not add a separate static torque and it does not create a position hold.
    joint4_angle_scale_raw = dict(payload.get("joint4_angle_dependent_gravity_scale", {}))
    joint4_base_scale = float(gravity_joint_scale.get(4, 1.0))
    joint4_angle_scale = {
        "enabled": bool(joint4_angle_scale_raw.get("enabled", False)),
        "q_start_rad": float(joint4_angle_scale_raw.get("q_start_rad", 0.45)),
        "q_full_rad": float(joint4_angle_scale_raw.get("q_full_rad", 0.80)),
        "scale_at_start": float(joint4_angle_scale_raw.get("scale_at_start", joint4_base_scale)),
        "scale_at_full": float(joint4_angle_scale_raw.get("scale_at_full", joint4_base_scale)),
        "blend": str(joint4_angle_scale_raw.get("blend", "smoothstep")),
        "raw": joint4_angle_scale_raw,
    }

    startup_ramp_sec = float(payload.get("startup_ramp_sec", 0.0))
    if startup_ramp_sec < 0.0 or startup_ramp_sec > 10.0:
        raise RuntimeError("startup_ramp_sec must be between 0 and 10 seconds")

    control_rate_hz = float(payload.get("control_rate_hz", 50.0))
    if control_rate_hz <= 0.0 or control_rate_hz > 150.0:
        raise RuntimeError("control_rate_hz must be >0 and <=150.0")

    for motor_id in target_joints:
        gravity_joint_scale.setdefault(motor_id, 1.0)
        zero_torque_kd.setdefault(motor_id, 0.03)
        max_abs_torque.setdefault(motor_id, 0.50)
        # Nm/s. 0 means no slew limiting.
        torque_slew_rate.setdefault(motor_id, 0.0)

        if gravity_joint_scale[motor_id] <= 0.0 or gravity_joint_scale[motor_id] > 3.0:
            raise RuntimeError(f"Joint{motor_id} gravity_joint_scale must be >0 and <=3.0")
        if zero_torque_kd[motor_id] < 0.0 or zero_torque_kd[motor_id] > 0.10:
            raise RuntimeError(f"Joint{motor_id} zero_torque_kd must be between 0 and 0.10")
        if max_abs_torque[motor_id] <= 0.0 or max_abs_torque[motor_id] > 7.0:
            raise RuntimeError(f"Joint{motor_id} max_abs_torque must be >0 and <=7.0")
        if torque_slew_rate[motor_id] < 0.0 or torque_slew_rate[motor_id] > 30.0:
            raise RuntimeError(f"Joint{motor_id} torque_slew_rate must be between 0 and 30 Nm/s")

        software_damping_coeff.setdefault(motor_id, 0.0)
        software_damping_max.setdefault(motor_id, 0.0)

        if software_damping_coeff[motor_id] < 0.0 or software_damping_coeff[motor_id] > 3.0:
            raise RuntimeError(f"Joint{motor_id} software damping must be between 0 and 3.0 Nm/(rad/s)")
        if software_damping_max[motor_id] < 0.0 or software_damping_max[motor_id] > 3.0:
            raise RuntimeError(f"Joint{motor_id} max_abs_damping_torque must be between 0 and 3.0 Nm")

    if qdot_filter_alpha <= 0.0 or qdot_filter_alpha > 1.0:
        raise RuntimeError("software_damping.qdot_filter_alpha must be >0 and <=1.0")
    if max_abs_qdot <= 0.0 or max_abs_qdot > 30.0:
        raise RuntimeError("software_damping.max_abs_qdot must be >0 and <=30.0 rad/s")

    if joint2_anti_j3_hold["q2_hold_mode"] != "capture_at_start":
        raise RuntimeError(
            "joint2_anti_j3_coupling_hold.q2_hold_mode currently supports only 'capture_at_start'"
        )
    if joint2_anti_j3_hold["j3_abs_torque_start"] < 0.0:
        raise RuntimeError("joint2_anti_j3_coupling_hold.j3_abs_torque_start must be >=0")
    if joint2_anti_j3_hold["j3_abs_torque_full"] <= joint2_anti_j3_hold["j3_abs_torque_start"]:
        raise RuntimeError(
            "joint2_anti_j3_coupling_hold.j3_abs_torque_full must be greater than j3_abs_torque_start"
        )
    if joint2_anti_j3_hold["kp"] < 0.0 or joint2_anti_j3_hold["kp"] > 5.0:
        raise RuntimeError("joint2_anti_j3_coupling_hold.kp must be between 0 and 5 Nm/rad")
    if joint2_anti_j3_hold["kd"] < 0.0 or joint2_anti_j3_hold["kd"] > 3.0:
        raise RuntimeError("joint2_anti_j3_coupling_hold.kd must be between 0 and 3 Nm/(rad/s)")
    if (
        joint2_anti_j3_hold["max_abs_hold_torque"] < 0.0
        or joint2_anti_j3_hold["max_abs_hold_torque"] > 3.0
    ):
        raise RuntimeError(
            "joint2_anti_j3_coupling_hold.max_abs_hold_torque must be between 0 and 3 Nm"
        )

    if joint4_angle_scale["blend"] not in ("linear", "smoothstep"):
        raise RuntimeError("joint4_angle_dependent_gravity_scale.blend must be 'linear' or 'smoothstep'")
    if joint4_angle_scale["q_full_rad"] <= joint4_angle_scale["q_start_rad"]:
        raise RuntimeError(
            "joint4_angle_dependent_gravity_scale.q_full_rad must be greater than q_start_rad"
        )
    if joint4_angle_scale["scale_at_start"] <= 0.0 or joint4_angle_scale["scale_at_start"] > 3.0:
        raise RuntimeError("joint4_angle_dependent_gravity_scale.scale_at_start must be >0 and <=3.0")
    if joint4_angle_scale["scale_at_full"] <= 0.0 or joint4_angle_scale["scale_at_full"] > 3.0:
        raise RuntimeError("joint4_angle_dependent_gravity_scale.scale_at_full must be >0 and <=3.0")

    software_damping = {
        "enabled": software_damping_enabled,
        "damping": software_damping_coeff,
        "max_abs_damping_torque": software_damping_max,
        "qdot_filter_alpha": qdot_filter_alpha,
        "max_abs_qdot": max_abs_qdot,
        "raw": software_damping_raw,
    }

    thermal_safety_raw = dict(payload.get("thermal_safety", {}))
    thermal_safety = {
        "enabled": bool(thermal_safety_raw.get("enabled", False)),
        "monitor_motor_ids": [int(x) for x in thermal_safety_raw.get("monitor_motor_ids", [])],
        "stop_temp_c": float(thermal_safety_raw.get("stop_temp_c", 85.0)),
        "max_feedback_age_sec": float(thermal_safety_raw.get("max_feedback_age_sec", 0.5)),
        "require_feedback_after_sec": float(thermal_safety_raw.get("require_feedback_after_sec", 3.0)),
        "consecutive_over_limit": int(thermal_safety_raw.get("consecutive_over_limit", 1)),
        "raw": thermal_safety_raw,
    }

    if thermal_safety["enabled"]:
        if not thermal_safety["monitor_motor_ids"]:
            raise RuntimeError("thermal_safety.monitor_motor_ids must not be empty when enabled")
        for motor_id in thermal_safety["monitor_motor_ids"]:
            if motor_id not in ALL_MOTOR_IDS:
                raise RuntimeError(f"thermal_safety.monitor_motor_ids contains invalid Joint{motor_id}")
        if thermal_safety["stop_temp_c"] <= 0.0 or thermal_safety["stop_temp_c"] > 145.0:
            raise RuntimeError("thermal_safety.stop_temp_c must be >0 and <=145 Celsius")
        if thermal_safety["max_feedback_age_sec"] <= 0.0 or thermal_safety["max_feedback_age_sec"] > 10.0:
            raise RuntimeError("thermal_safety.max_feedback_age_sec must be >0 and <=10 seconds")
        if thermal_safety["require_feedback_after_sec"] < 0.0 or thermal_safety["require_feedback_after_sec"] > 30.0:
            raise RuntimeError("thermal_safety.require_feedback_after_sec must be between 0 and 30 seconds")
        if thermal_safety["consecutive_over_limit"] < 1 or thermal_safety["consecutive_over_limit"] > 100:
            raise RuntimeError("thermal_safety.consecutive_over_limit must be between 1 and 100")

    return {
        "target_joints": target_joints,
        "gravity_feedforward_ratio": gravity_feedforward_ratio,
        "gravity_joint_scale": gravity_joint_scale,
        "zero_torque_kd": zero_torque_kd,
        "max_abs_torque": max_abs_torque,
        "torque_slew_rate": torque_slew_rate,
        "software_damping": software_damping,
        "thermal_safety": thermal_safety,
        "joint2_anti_j3_coupling_hold": joint2_anti_j3_hold,
        "joint4_angle_dependent_gravity_scale": joint4_angle_scale,
        "startup_ramp_sec": startup_ramp_sec,
        "control_rate_hz": control_rate_hz,
        "raw": payload,
    }


def load_pinocchio_model(urdf_path: Path):
    if not urdf_path.exists():
        raise RuntimeError(f"URDF not found: {urdf_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    joint_index = {}
    for motor_id in ARM_JOINT_IDS:
        joint_name = f"Joint{motor_id}"
        jid = model.getJointId(joint_name)
        if jid >= model.njoints:
            raise RuntimeError(f"Pinocchio joint not found in URDF: {joint_name}")

        idx_q = model.idx_qs[jid]
        idx_v = model.idx_vs[jid]
        joint_index[motor_id] = (jid, idx_q, idx_v)

    return model, data, joint_index


def build_pinocchio_q(model, joint_index, q_urdf: Dict[int, float]):
    q = pin.neutral(model)

    for motor_id in ARM_JOINT_IDS:
        joint_name = f"Joint{motor_id}"
        jid, idx_q, _idx_v = joint_index[motor_id]
        nq = model.nqs[jid]
        nv = model.nvs[jid]
        theta = float(q_urdf[motor_id])

        if nq == 1 and nv == 1:
            q[idx_q] = theta
        elif nq == 2 and nv == 1:
            # Continuous revolute joint in Pinocchio uses [cos(theta), sin(theta)].
            q[idx_q] = np.cos(theta)
            q[idx_q + 1] = np.sin(theta)
        else:
            raise RuntimeError(
                f"{joint_name} has nq={nq}, nv={nv}; expected revolute nq=1,nv=1 or continuous nq=2,nv=1."
            )

    return q


def motor_pos_to_q_urdf(motor_pos_map: Dict[int, float], sign, zero):
    q_urdf: Dict[int, float] = {}
    for motor_id in ARM_JOINT_IDS:
        q_urdf[motor_id] = sign[motor_id] * (motor_pos_map[motor_id] - zero[motor_id])
    return q_urdf


def compute_gravity_tau(model, data, joint_index, q_urdf: Dict[int, float]):
    q = build_pinocchio_q(model, joint_index, q_urdf)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)

    tau_vec = pin.rnea(model, data, q, v, a)

    tau_urdf: Dict[int, float] = {}
    for motor_id in ARM_JOINT_IDS:
        _jid, _idx_q, idx_v = joint_index[motor_id]
        tau_urdf[motor_id] = float(tau_vec[idx_v])

    return tau_urdf


def smoothstep01(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_angle_dependent_scale(
    effective: Dict[int, float],
    q_urdf: Dict[int, float],
    motor_id: int,
    angle_scale_cfg,
):
    if motor_id not in effective:
        return
    if not angle_scale_cfg:
        return
    if not angle_scale_cfg.get("enabled", False):
        return

    q = float(q_urdf.get(motor_id, 0.0))
    q_start = float(angle_scale_cfg["q_start_rad"])
    q_full = float(angle_scale_cfg["q_full_rad"])
    scale_start = float(angle_scale_cfg["scale_at_start"])
    scale_full = float(angle_scale_cfg["scale_at_full"])

    t = clamp((q - q_start) / (q_full - q_start), 0.0, 1.0)
    if angle_scale_cfg.get("blend", "smoothstep") == "smoothstep":
        t = smoothstep01(t)

    effective[motor_id] = scale_start + t * (scale_full - scale_start)


def compute_effective_gravity_joint_scales(
    q_urdf: Dict[int, float],
    target_joints: List[int],
    gravity_joint_scale: Dict[int, float],
    joint4_angle_dependent_gravity_scale=None,
):
    effective = {mid: float(gravity_joint_scale[mid]) for mid in target_joints}

    apply_angle_dependent_scale(
        effective=effective,
        q_urdf=q_urdf,
        motor_id=4,
        angle_scale_cfg=joint4_angle_dependent_gravity_scale,
    )

    return effective


def compute_target_motor_torque(
    tau_urdf: Dict[int, float],
    sign: Dict[int, float],
    target_joints: List[int],
    gravity_feedforward_ratio: float,
    gravity_joint_scale: Dict[int, float],
    q_urdf: Dict[int, float] = None,
    joint4_angle_dependent_gravity_scale=None,
):
    q_urdf = q_urdf or {}
    effective_scale = compute_effective_gravity_joint_scales(
        q_urdf=q_urdf,
        target_joints=target_joints,
        gravity_joint_scale=gravity_joint_scale,
        joint4_angle_dependent_gravity_scale=joint4_angle_dependent_gravity_scale,
    )

    torque_motor: Dict[int, float] = {}
    for motor_id in target_joints:
        torque_motor[motor_id] = (
            sign[motor_id]
            * tau_urdf[motor_id]
            * gravity_feedforward_ratio
            * effective_scale[motor_id]
        )
    return torque_motor


def apply_startup_ramp(
    target_torque: Dict[int, float],
    target_joints: List[int],
    alpha: float,
):
    return {mid: alpha * target_torque[mid] for mid in target_joints}


def apply_torque_slew_limit(
    desired_torque: Dict[int, float],
    prev_torque: Dict[int, float],
    target_joints: List[int],
    torque_slew_rate: Dict[int, float],
    dt: float,
):
    out: Dict[int, float] = {}

    for mid in target_joints:
        desired = desired_torque[mid]
        prev = prev_torque.get(mid, 0.0)
        rate = torque_slew_rate.get(mid, 0.0)

        if rate <= 0.0:
            out[mid] = desired
            continue

        max_delta = rate * max(0.0, dt)
        delta = desired - prev

        if delta > max_delta:
            out[mid] = prev + max_delta
        elif delta < -max_delta:
            out[mid] = prev - max_delta
        else:
            out[mid] = desired

    return out


def torque_within_limits(torque_motor: Dict[int, float], max_abs_torque: Dict[int, float]):
    for motor_id, torque in torque_motor.items():
        limit = max_abs_torque[motor_id]
        if abs(torque) > limit:
            return False, f"Joint{motor_id} torque {torque:+.6f} Nm exceeds limit {limit:.6f} Nm"
    return True, "OK"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_qdot_urdf(
    q_urdf: Dict[int, float],
    prev_q_urdf,
    prev_qdot_urdf: Dict[int, float],
    target_joints: List[int],
    dt: float,
    qdot_filter_alpha: float,
    max_abs_qdot: float,
):
    if prev_q_urdf is None or dt <= 1e-6:
        zero = {mid: 0.0 for mid in target_joints}
        return zero, zero

    raw_qdot: Dict[int, float] = {}
    filtered_qdot: Dict[int, float] = {}

    for mid in target_joints:
        dq = float(q_urdf[mid] - prev_q_urdf[mid])
        # Adjacent samples are close at 100 Hz, but wrap protection avoids one bad diff.
        if dq > np.pi:
            dq -= 2.0 * np.pi
        elif dq < -np.pi:
            dq += 2.0 * np.pi

        qdot = clamp(dq / dt, -max_abs_qdot, max_abs_qdot)
        old = float(prev_qdot_urdf.get(mid, 0.0))
        filt = qdot_filter_alpha * qdot + (1.0 - qdot_filter_alpha) * old

        raw_qdot[mid] = qdot
        filtered_qdot[mid] = filt

    return raw_qdot, filtered_qdot


def compute_software_damping_motor_torque(
    qdot_urdf: Dict[int, float],
    sign: Dict[int, float],
    target_joints: List[int],
    software_damping,
):
    damping_motor: Dict[int, float] = {}

    if not software_damping.get("enabled", False):
        return {mid: 0.0 for mid in target_joints}

    damping_coeff = software_damping["damping"]
    damping_limit = software_damping["max_abs_damping_torque"]

    for mid in target_joints:
        # URDF-side viscous damping: tau_damp_urdf = -D * qdot_urdf.
        tau_damp_urdf = -float(damping_coeff[mid]) * float(qdot_urdf.get(mid, 0.0))
        # Convert URDF torque sign to motor torque sign using the same convention as gravity_ff.
        tau_damp_motor = float(sign[mid]) * tau_damp_urdf

        limit = float(damping_limit.get(mid, 0.0))
        if limit > 0.0:
            tau_damp_motor = clamp(tau_damp_motor, -limit, limit)

        damping_motor[mid] = tau_damp_motor

    return damping_motor


def compute_joint2_anti_j3_coupling_hold_motor_torque(
    q_urdf: Dict[int, float],
    qdot_urdf: Dict[int, float],
    target_joints: List[int],
    reference_torque_motor: Dict[int, float],
    joint2_anti_j3_hold,
    q2_hold_ref,
):
    """
    Joint2 anti-J3-coupling hold term, returned as MOTOR-SIDE torque.

    Purpose:
      Joint2-only can stay stable, but Joint3 large torque/velocity can pull Joint2 away.
      This term adds a small, gated virtual hold torque on Joint2 only when Joint3 output
      is large enough.

    Convention for this robot:
      - Joint2 positive Type1 motor torque tends to decrease q_urdf.
      - Therefore:
          if q2 > q2_hold, positive motor torque helps pull q2 back down.
          if q2 < q2_hold, negative motor torque helps push q2 back up.

      Motor-side hold formula:
          tau_hold_motor = kp * (q2 - q2_hold) + kd * qdot2

      The term is gated by |Joint3 reference motor torque|, so it stays near zero
      when Joint3 is not strongly involved.
    """
    hold_motor = {mid: 0.0 for mid in target_joints}

    if 2 not in target_joints:
        return hold_motor
    if 3 not in target_joints:
        return hold_motor
    if not joint2_anti_j3_hold.get("enabled", False):
        return hold_motor
    if q2_hold_ref is None:
        return hold_motor

    q2 = float(q_urdf.get(2, 0.0))
    qdot2 = float(qdot_urdf.get(2, 0.0))

    j3_abs = abs(float(reference_torque_motor.get(3, 0.0)))
    j3_start = float(joint2_anti_j3_hold["j3_abs_torque_start"])
    j3_full = float(joint2_anti_j3_hold["j3_abs_torque_full"])

    gate = clamp((j3_abs - j3_start) / (j3_full - j3_start), 0.0, 1.0)
    if gate <= 0.0:
        return hold_motor

    kp = float(joint2_anti_j3_hold["kp"])
    kd = float(joint2_anti_j3_hold["kd"])
    max_abs_hold_torque = float(joint2_anti_j3_hold["max_abs_hold_torque"])

    hold = gate * (kp * (q2 - float(q2_hold_ref)) + kd * qdot2)

    if max_abs_hold_torque > 0.0:
        hold = clamp(hold, -max_abs_hold_torque, max_abs_hold_torque)

    hold_motor[2] = hold
    return hold_motor


def add_torque_terms(
    base_torque: Dict[int, float],
    extra_torque: Dict[int, float],
    target_joints: List[int],
):
    return {mid: float(base_torque[mid]) + float(extra_torque.get(mid, 0.0)) for mid in target_joints}


def read_all_feedback_precheck(driver: SukineeSocketCANDriver):
    values, statuses = driver.read_many_params_float(
        motor_ids=ALL_MOTOR_IDS,
        params=PARAMS_ALL,
        timeout=0.5,
        inter_request_delay=0.003,
    )
    return values, statuses


def feedback_ok(values, statuses) -> Tuple[bool, str]:
    for motor_id in ALL_MOTOR_IDS:
        for _index, name in PARAMS_ALL:
            status = statuses.get(motor_id, {}).get(name)
            if status != "OK":
                return False, f"Joint{motor_id} {name} status={status}"

        vbus = float(values[motor_id]["vbus"])
        iqf = float(values[motor_id]["iqf"])
        vel = float(values[motor_id]["vel"])

        if not (40.0 <= vbus <= 55.0):
            return False, f"Joint{motor_id} vbus out of range: {vbus:.3f} V"
        if abs(iqf) > 0.5:
            return False, f"Joint{motor_id} iqf too large before mode: {iqf:.3f} A"
        if abs(vel) > 0.8:
            return False, f"Joint{motor_id} vel too large before mode: {vel:.3f} rad/s"

    return True, "OK"


def check_thermal_safety(driver: SukineeSocketCANDriver, thermal_safety, mode_elapsed: float, over_count):
    """Raise RuntimeError if monitored motor temperature exceeds stop_temp_c.

    Temperature source: latest cached Type2 feedback.
    RobStride private-protocol Type2 feedback defines Byte6~7 as Temp(Celsius) * 10.
    """
    if not thermal_safety.get("enabled", False):
        return {}

    monitor_motor_ids = [int(x) for x in thermal_safety["monitor_motor_ids"]]
    stop_temp_c = float(thermal_safety["stop_temp_c"])
    max_age = float(thermal_safety["max_feedback_age_sec"])
    require_after = float(thermal_safety["require_feedback_after_sec"])
    consecutive_limit = int(thermal_safety["consecutive_over_limit"])

    latest_temps = {}
    missing = []

    for motor_id in monitor_motor_ids:
        fb = driver.get_latest_type2_feedback(motor_id, max_age=max_age)
        if fb is None:
            missing.append(motor_id)
            continue

        temp_c = float(fb.temperature)
        latest_temps[motor_id] = temp_c

        if temp_c >= stop_temp_c:
            over_count[motor_id] = int(over_count.get(motor_id, 0)) + 1
        else:
            over_count[motor_id] = 0

        if over_count[motor_id] >= consecutive_limit:
            raise RuntimeError(
                f"THERMAL SAFETY STOP: Joint{motor_id} temperature {temp_c:.1f} C "
                f">= stop_temp_c {stop_temp_c:.1f} C "
                f"for {over_count[motor_id]} consecutive check(s)"
            )

    if missing and mode_elapsed >= require_after:
        missing_text = ", ".join([f"Joint{mid}" for mid in missing])
        raise RuntimeError(
            f"THERMAL SAFETY STOP: missing fresh Type2 temperature feedback for "
            f"{missing_text} after {mode_elapsed:.2f} s; max_feedback_age_sec={max_age:.2f}"
        )

    return latest_temps


def read_arm_positions_fast(driver: SukineeSocketCANDriver, timeout: float):
    motor_pos_map: Dict[int, float] = {}
    for motor_id in ARM_JOINT_IDS:
        status, value = driver.read_param_float(motor_id, POS_INDEX, timeout=timeout)
        if status != "OK":
            return False, motor_pos_map, f"Joint{motor_id} pos status={status}"
        motor_pos_map[motor_id] = float(value)
    return True, motor_pos_map, "OK"


def send_disable_targets(driver: SukineeSocketCANDriver, target_joints: List[int], delay: float = 0.002):
    for motor_id in target_joints:
        driver.send_disable(motor_id, clear_fault=False)
        time.sleep(delay)


def send_set_motion_mode_targets(driver: SukineeSocketCANDriver, target_joints: List[int], delay: float = 0.002):
    for motor_id in target_joints:
        driver.send_set_motion_mode(motor_id)
        time.sleep(delay)


def send_enable_targets(driver: SukineeSocketCANDriver, target_joints: List[int], delay: float = 0.002):
    for motor_id in target_joints:
        driver.send_enable(motor_id)
        time.sleep(delay)


def send_type1_targets(
    driver: SukineeSocketCANDriver,
    target_joints: List[int],
    motor_pos_map: Dict[int, float],
    torque_motor: Dict[int, float],
    zero_torque_kd: Dict[int, float],
    delay: float = 0.0,
):
    for motor_id in target_joints:
        driver.send_motion_control(
            motor_id=motor_id,
            position=float(motor_pos_map[motor_id]),
            velocity=0.0,
            kp=0.0,
            kd=float(zero_torque_kd[motor_id]),
            torque=float(torque_motor[motor_id]),
        )
        if delay > 0:
            time.sleep(delay)


class ZeroTorqueJointStatePublisher:
    """Lightweight /joint_states publisher for RViz sync during zero-torque mode.

    ROS 2 imports are intentionally lazy, so the controller can still be
    imported or syntax-checked in a non-ROS shell when joint-state publishing
    is not requested.
    """

    def __init__(self):
        try:
            import rclpy
            from sensor_msgs.msg import JointState
        except Exception as exc:
            raise RuntimeError(
                "Failed to import ROS 2 Python packages. "
                "Run: source /opt/ros/jazzy/setup.bash && source install/setup.bash"
            ) from exc

        self.rclpy = rclpy
        self.JointState = JointState
        self._owns_context = False

        if not self.rclpy.ok():
            self.rclpy.init(args=None)
            self._owns_context = True

        self.node = self.rclpy.create_node("sukinee_zero_torque_joint_state_publisher")
        self.pub = self.node.create_publisher(JointState, "/joint_states", 1)

    def publish_q_urdf(self, q_urdf: Dict[int, float]):
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = JOINT_STATE_NAMES

        # Joint7 is published as 0.0 for now, matching the existing real
        # feedback publisher. Joint7 is not part of arm gravity compensation.
        msg.position = [
            float(q_urdf.get(1, 0.0)),
            float(q_urdf.get(2, 0.0)),
            float(q_urdf.get(3, 0.0)),
            float(q_urdf.get(4, 0.0)),
            float(q_urdf.get(5, 0.0)),
            float(q_urdf.get(6, 0.0)),
            0.0,
        ]

        self.pub.publish(msg)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        try:
            self.node.destroy_node()
        finally:
            if self._owns_context and self.rclpy.ok():
                self.rclpy.shutdown()


def print_preview(q_urdf, tau_urdf, target_torque, cfg):
    target_joints = cfg["target_joints"]
    gravity_joint_scale = cfg["gravity_joint_scale"]
    zero_torque_kd = cfg["zero_torque_kd"]
    max_abs_torque = cfg["max_abs_torque"]
    torque_slew_rate = cfg["torque_slew_rate"]
    monitor_joints = [mid for mid in MONITOR_JOINT_IDS if mid in target_joints]

    print("Current q_urdf and initial motor torque preview (Joint2-Joint4 only):")
    print("Joint | q_urdf(rad) | q_urdf(deg) | gain | motor_kd | slew(Nm/s) | max_abs | tau_urdf | target_ff")
    print("------------------------------------------------------------------------------------------------------")

    effective_scale_for_preview = compute_effective_gravity_joint_scales(
        q_urdf=q_urdf,
        target_joints=target_joints,
        gravity_joint_scale=gravity_joint_scale,
        joint4_angle_dependent_gravity_scale=cfg.get("joint4_angle_dependent_gravity_scale", None),
    )

    for motor_id in monitor_joints:
        tau = tau_urdf[motor_id]
        q = float(q_urdf[motor_id])
        print(
            f"J{motor_id:<4} | "
            f"{q:+.6f} | "
            f"{np.degrees(q):+.2f} | "
            f"{effective_scale_for_preview[motor_id]:.3f} | "
            f"{zero_torque_kd[motor_id]:.4f} | "
            f"{torque_slew_rate[motor_id]:.3f} | "
            f"{max_abs_torque[motor_id]:.3f} | "
            f"{tau:+.6f} | "
            f"{target_torque[motor_id]:+.6f}"
        )

    max_abs_cmd = max(abs(target_torque[mid]) for mid in target_joints) if target_joints else 0.0
    print()
    print(f"Max abs target motor_ff command over all controlled joints: {max_abs_cmd:.6f} Nm")
    print(f"startup_ramp_sec: {cfg['startup_ramp_sec']:.3f} s")
    print("Note: terminal preview is limited to Joint2-Joint4; control target_joints still follows config JSON.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sukinee SocketCAN zero-torque gravity mode with startup ramp and torque slew limit."
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--offset-json", default=DEFAULT_OFFSET_JSON)
    parser.add_argument("--config-json", default=DEFAULT_CONFIG_JSON)


    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds. 0 means run until Ctrl+C. Suggested: 10.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Override config control_rate_hz. 0 means use config.",
    )
    parser.add_argument(
        "--pos-timeout",
        type=float,
        default=0.03,
        help="Timeout for each Joint1-Joint6 pos Type17 read in control loop.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print status every N cycles.",
    )
    parser.add_argument(
        "--type1-delay",
        type=float,
        default=0.0,
        help="Optional delay between Type1 frames. Default 0.",
    )
    parser.add_argument(
        "--publish-joint-states",
        action="store_true",
        help="Publish /joint_states from the zero-torque loop for RViz sync.",
    )
    parser.add_argument(
        "--joint-state-every",
        type=int,
        default=2,
        help="Publish /joint_states every N control cycles. Default 2 is about 50 Hz at 100 Hz control.",
    )

    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm", default="")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Sukinee SocketCAN zero-torque gravity mode")
    print("Mode: startup ramp + torque slew limit + software damping + optional Joint2 anti-J3 hold")
    print()
    print("Safety status:")
    print("  This script can send REAL Type1 torque_ff commands when armed.")
    print("  Native SocketCAN driver is used. No cansend/candump subprocess.")
    print("  Kp = 0.0")
    print("  Kd comes from config JSON per joint.")
    print("  Gravity feedforward ratio and joint scale come from config JSON.")
    print("  startup_ramp_sec and torque_slew_rate come from config JSON.")
    print("  Optional joint2_anti_j3_coupling_hold adds a gated Joint2 virtual hold term H2 when Joint3 output is large.")
    print("  Optional joint4_angle_dependent_gravity_scale changes only Joint4 gravity gain as a function of q4.")
    print("  Joint7 is never commanded.")
    print("  finally path sends Type4 disable to target joints.")
    print("  Dry-run mode sends NO Type3 / Type1 / Type4 command.")
    print("  Optional /joint_states publishing uses existing q_urdf from the control loop; it does not add CAN reads.")
    print()

    offset_path = Path(args.offset_json)
    config_path = Path(args.config_json)
    urdf_path = Path(args.urdf)
    sign, zero, _offset_payload = load_offset_json(offset_path)
    cfg = load_gravity_config(config_path)

    if args.rate > 0.0:
        if args.rate > 150.0:
            print("RESULT: FAIL")
            print("ERROR: --rate override must be <=150.0 Hz.")
            return 2
        cfg["control_rate_hz"] = float(args.rate)

    if args.duration < 0.0 or args.duration > 120.0:
        print("RESULT: FAIL")
        print("ERROR: --duration must be >=0 and <=120 seconds.")
        return 2

    if args.pos_timeout <= 0.0 or args.pos_timeout > 0.5:
        print("RESULT: FAIL")
        print("ERROR: --pos-timeout must be >0 and <=0.5 seconds.")
        return 2

    if args.joint_state_every <= 0:
        print("RESULT: FAIL")
        print("ERROR: --joint-state-every must be >= 1.")
        return 2

    model, data, joint_index = load_pinocchio_model(urdf_path)

    target_joints = cfg["target_joints"]
    control_rate_hz = cfg["control_rate_hz"]

    print("Loaded files:")
    print(f"  URDF:                    {urdf_path}")
    print(f"  offset JSON:             {offset_path}")
    print(f"  config JSON:             {config_path}")
    print()

    print("Target joints:")
    print(f"  {target_joints}")
    print(f"Control rate: {control_rate_hz:.2f} Hz")
    print(f"Duration:     {'until Ctrl+C' if args.duration == 0.0 else f'{args.duration:.3f} s'}")
    print(f"pos timeout:  {args.pos_timeout:.3f} s")
    print(f"publish /joint_states: {args.publish_joint_states}")
    if args.publish_joint_states:
        print(f"joint state every: {args.joint_state_every} cycle(s)")
        print("Do not run sukinee_real_feedback_joint_state_publisher.py at the same time.")
    print()

    print("Loaded motor-to-URDF mapping shown for terminal monitor joints only:")
    for motor_id in MONITOR_JOINT_IDS:
        print(f"  Joint{motor_id}: sign={sign[motor_id]:+.1f}, zero={zero[motor_id]:+.9f}")
    print()

    print("Loaded gravity config:")
    print(f"  gravity_feedforward_ratio = {cfg['gravity_feedforward_ratio']:.3f}")
    print(f"  startup_ramp_sec = {cfg['startup_ramp_sec']:.3f}")
    print(f"  software_damping.enabled = {cfg['software_damping']['enabled']}")
    if cfg["software_damping"]["enabled"]:
        print(f"  software_damping.qdot_filter_alpha = {cfg['software_damping']['qdot_filter_alpha']:.3f}")
        print(f"  software_damping.max_abs_qdot = {cfg['software_damping']['max_abs_qdot']:.3f} rad/s")
    j2h = cfg["joint2_anti_j3_coupling_hold"]
    print(f"  joint2_anti_j3_coupling_hold.enabled = {j2h['enabled']}")
    if j2h["enabled"]:
        print(f"  joint2_anti_j3_coupling_hold.q2_hold_mode = {j2h['q2_hold_mode']}")
        print(f"  joint2_anti_j3_coupling_hold.j3_abs_torque_start = {j2h['j3_abs_torque_start']:.3f} Nm")
        print(f"  joint2_anti_j3_coupling_hold.j3_abs_torque_full = {j2h['j3_abs_torque_full']:.3f} Nm")
        print(f"  joint2_anti_j3_coupling_hold.kp = {j2h['kp']:.3f} Nm/rad")
        print(f"  joint2_anti_j3_coupling_hold.kd = {j2h['kd']:.3f} Nm/(rad/s)")
        print(f"  joint2_anti_j3_coupling_hold.max_abs_hold_torque = {j2h['max_abs_hold_torque']:.3f} Nm")
    j4s = cfg["joint4_angle_dependent_gravity_scale"]
    print(f"  joint4_angle_dependent_gravity_scale.enabled = {j4s['enabled']}")
    if j4s["enabled"]:
        print(f"  joint4_angle_dependent_gravity_scale.q_start_rad = {j4s['q_start_rad']:.3f}")
        print(f"  joint4_angle_dependent_gravity_scale.q_full_rad = {j4s['q_full_rad']:.3f}")
        print(f"  joint4_angle_dependent_gravity_scale.scale_at_start = {j4s['scale_at_start']:.3f}")
        print(f"  joint4_angle_dependent_gravity_scale.scale_at_full = {j4s['scale_at_full']:.3f}")
        print(f"  joint4_angle_dependent_gravity_scale.blend = {j4s['blend']}")
    ts = cfg["thermal_safety"]
    print(f"  thermal_safety.enabled = {ts['enabled']}")
    if ts["enabled"]:
        print(f"  thermal_safety.monitor_motor_ids = {ts['monitor_motor_ids']}")
        print(f"  thermal_safety.stop_temp_c = {ts['stop_temp_c']:.1f} C")
        print(f"  thermal_safety.max_feedback_age_sec = {ts['max_feedback_age_sec']:.3f} s")
        print(f"  thermal_safety.require_feedback_after_sec = {ts['require_feedback_after_sec']:.3f} s")
        print(f"  thermal_safety.consecutive_over_limit = {ts['consecutive_over_limit']}")
    print("  Per-joint config shown for terminal monitor joints only:")
    for motor_id in [mid for mid in MONITOR_JOINT_IDS if mid in target_joints]:
        sd = cfg["software_damping"]
        print(
            f"  Joint{motor_id}: "
            f"scale={cfg['gravity_joint_scale'][motor_id]:.3f}, "
            f"motor_kd={cfg['zero_torque_kd'][motor_id]:.4f}, "
            f"softD={sd['damping'][motor_id]:.4f} Nm/(rad/s), "
            f"softDmax={sd['max_abs_damping_torque'][motor_id]:.3f} Nm, "
            f"slew={cfg['torque_slew_rate'][motor_id]:.3f} Nm/s, "
            f"max_abs_torque={cfg['max_abs_torque'][motor_id]:.3f}"
        )
    print()

    driver = SukineeSocketCANDriver(args.can)
    did_send_real_command = False
    joint_state_publisher = None

    try:
        if args.publish_joint_states:
            joint_state_publisher = ZeroTorqueJointStatePublisher()

        driver.open()

        print("=" * 90)
        print("Pre-check: Type17 read-only feedback")
        print("=" * 90)

        values, statuses = read_all_feedback_precheck(driver)
        ok, reason = feedback_ok(values, statuses)
        if not ok:
            print("RESULT: FAIL")
            print(f"Feedback pre-check failed: {reason}")
            return 1

        motor_pos_map = {mid: float(values[mid]["pos"]) for mid in ARM_JOINT_IDS}
        q_urdf = motor_pos_to_q_urdf(motor_pos_map, sign, zero)
        tau_urdf = compute_gravity_tau(model, data, joint_index, q_urdf)
        gravity_torque_preview = compute_target_motor_torque(
            tau_urdf=tau_urdf,
            sign=sign,
            target_joints=target_joints,
            gravity_feedforward_ratio=cfg["gravity_feedforward_ratio"],
            gravity_joint_scale=cfg["gravity_joint_scale"],
            q_urdf=q_urdf,
            joint4_angle_dependent_gravity_scale=cfg["joint4_angle_dependent_gravity_scale"],
        )
        zero_qdot_preview = {mid: 0.0 for mid in target_joints}
        damping_torque_preview = compute_software_damping_motor_torque(
            qdot_urdf=zero_qdot_preview,
            sign=sign,
            target_joints=target_joints,
            software_damping=cfg["software_damping"],
        )
        target_torque = add_torque_terms(gravity_torque_preview, damping_torque_preview, target_joints)
        joint2_anti_j3_hold_preview = compute_joint2_anti_j3_coupling_hold_motor_torque(
            q_urdf=q_urdf,
            qdot_urdf=zero_qdot_preview,
            target_joints=target_joints,
            reference_torque_motor=target_torque,
            joint2_anti_j3_hold=cfg["joint2_anti_j3_coupling_hold"],
            q2_hold_ref=q_urdf.get(2, None),
        )
        target_torque = add_torque_terms(target_torque, joint2_anti_j3_hold_preview, target_joints)

        ok, reason = torque_within_limits(target_torque, cfg["max_abs_torque"])
        print_preview(q_urdf, tau_urdf, target_torque, cfg)

        if joint_state_publisher is not None:
            joint_state_publisher.publish_q_urdf(q_urdf)

        if not ok:
            print("RESULT: FAIL")
            print(f"Initial torque check failed: {reason}")
            return 1

        if not args.armed:
            print()
            print("DRY-RUN ONLY. No Type3 / Type1 / Type4 command was sent.")
            print("To actually run the SocketCAN zero-torque gravity mode:")
            print(f"  --armed --confirm {ARMED_CONFIRM_TEXT}")
            print()
            print("RESULT: DRY_RUN_PASS")
            return 0

        if args.confirm != ARMED_CONFIRM_TEXT:
            print("RESULT: FAIL")
            print("ERROR: armed mode requires exact confirmation:")
            print(f"  --confirm {ARMED_CONFIRM_TEXT}")
            return 2

        did_send_real_command = True

        print()
        print("=" * 90)
        print("ARMED: entering SocketCAN zero-torque gravity mode")
        print("=" * 90)
        print("Keep one hand on the arm. Press Ctrl+C immediately if direction feels wrong.")
        print()

        send_disable_targets(driver, target_joints)
        time.sleep(0.05)

        send_set_motion_mode_targets(driver, target_joints)
        time.sleep(0.05)

        send_enable_targets(driver, target_joints)
        time.sleep(0.05)

        period = 1.0 / control_rate_hz

        # IMPORTANT:
        # Start loop timing AFTER all disable/mode/enable setup commands.
        # The previous pasted version had the print/cycle/sleep block outside the
        # while loop, so duration elapsed while cycle stayed at 0. This version keeps
        # printing, cycle increment, and sleep inside the real-time loop.
        loop_start_time = time.monotonic()
        start_time = loop_start_time
        last_time = loop_start_time
        deadline = None if args.duration == 0.0 else loop_start_time + args.duration

        cycle = 0
        overrun_count = 0
        read_fail_count = 0
        max_cycle_time = 0.0
        cycle_times = []

        prev_torque = {mid: 0.0 for mid in target_joints}
        prev_q_urdf = None
        prev_qdot_urdf = {mid: 0.0 for mid in target_joints}
        max_abs_qdot_seen = {mid: 0.0 for mid in target_joints}
        max_abs_damping_seen = {mid: 0.0 for mid in target_joints}
        max_abs_joint2_anti_j3_hold_seen = 0.0
        q2_hold_ref = None
        thermal_over_count = {mid: 0 for mid in cfg["thermal_safety"].get("monitor_motor_ids", [])}
        latest_thermal_temps = {}

        while True:
            cycle_start = time.monotonic()
            if deadline is not None and cycle_start >= deadline:
                break

            dt_since_last = cycle_start - last_time
            last_time = cycle_start

            ok, motor_pos_map, read_reason = read_arm_positions_fast(
                driver=driver,
                timeout=args.pos_timeout,
            )
            if not ok:
                read_fail_count += 1
                raise RuntimeError(f"Arm position read failed during mode: {read_reason}")

            q_urdf = motor_pos_to_q_urdf(motor_pos_map, sign, zero)

            if joint_state_publisher is not None and cycle % args.joint_state_every == 0:
                joint_state_publisher.publish_q_urdf(q_urdf)

            if (
                q2_hold_ref is None
                and 2 in target_joints
                and cfg["joint2_anti_j3_coupling_hold"].get("enabled", False)
            ):
                q2_hold_ref = float(q_urdf[2])
            tau_urdf = compute_gravity_tau(model, data, joint_index, q_urdf)

            gravity_torque = compute_target_motor_torque(
                tau_urdf=tau_urdf,
                sign=sign,
                target_joints=target_joints,
                gravity_feedforward_ratio=cfg["gravity_feedforward_ratio"],
                gravity_joint_scale=cfg["gravity_joint_scale"],
                q_urdf=q_urdf,
                joint4_angle_dependent_gravity_scale=cfg["joint4_angle_dependent_gravity_scale"],
            )

            _raw_qdot_urdf, qdot_urdf = estimate_qdot_urdf(
                q_urdf=q_urdf,
                prev_q_urdf=prev_q_urdf,
                prev_qdot_urdf=prev_qdot_urdf,
                target_joints=target_joints,
                dt=dt_since_last,
                qdot_filter_alpha=cfg["software_damping"]["qdot_filter_alpha"],
                max_abs_qdot=cfg["software_damping"]["max_abs_qdot"],
            )

            damping_torque = compute_software_damping_motor_torque(
                qdot_urdf=qdot_urdf,
                sign=sign,
                target_joints=target_joints,
                software_damping=cfg["software_damping"],
            )

            target_torque = add_torque_terms(gravity_torque, damping_torque, target_joints)
            joint2_anti_j3_hold_torque = compute_joint2_anti_j3_coupling_hold_motor_torque(
                q_urdf=q_urdf,
                qdot_urdf=qdot_urdf,
                target_joints=target_joints,
                reference_torque_motor=target_torque,
                joint2_anti_j3_hold=cfg["joint2_anti_j3_coupling_hold"],
                q2_hold_ref=q2_hold_ref,
            )
            target_torque = add_torque_terms(target_torque, joint2_anti_j3_hold_torque, target_joints)

            for mid in target_joints:
                max_abs_qdot_seen[mid] = max(max_abs_qdot_seen[mid], abs(qdot_urdf[mid]))
                max_abs_damping_seen[mid] = max(max_abs_damping_seen[mid], abs(damping_torque[mid]))

            if 2 in target_joints:
                max_abs_joint2_anti_j3_hold_seen = max(
                    max_abs_joint2_anti_j3_hold_seen,
                    abs(float(joint2_anti_j3_hold_torque.get(2, 0.0))),
                )

            prev_q_urdf = dict(q_urdf)
            prev_qdot_urdf = dict(qdot_urdf)

            ramp_sec = cfg["startup_ramp_sec"]
            if ramp_sec > 0.0:
                ramp_alpha = min(1.0, max(0.0, (cycle_start - start_time) / ramp_sec))
            else:
                ramp_alpha = 1.0

            ramped_torque = apply_startup_ramp(target_torque, target_joints, ramp_alpha)
            torque_motor = apply_torque_slew_limit(
                desired_torque=ramped_torque,
                prev_torque=prev_torque,
                target_joints=target_joints,
                torque_slew_rate=cfg["torque_slew_rate"],
                dt=dt_since_last,
            )
            prev_torque = dict(torque_motor)

            ok, reason = torque_within_limits(torque_motor, cfg["max_abs_torque"])
            if not ok:
                raise RuntimeError(f"Torque limit exceeded during mode: {reason}")

            latest_thermal_temps = check_thermal_safety(
                driver=driver,
                thermal_safety=cfg["thermal_safety"],
                mode_elapsed=cycle_start - loop_start_time,
                over_count=thermal_over_count,
            )

            send_type1_targets(
                driver=driver,
                target_joints=target_joints,
                motor_pos_map=motor_pos_map,
                torque_motor=torque_motor,
                zero_torque_kd=cfg["zero_torque_kd"],
                delay=args.type1_delay,
            )

            elapsed = time.monotonic() - cycle_start
            cycle_times.append(elapsed)
            max_cycle_time = max(max_cycle_time, elapsed)

            if cycle % max(1, args.print_every) == 0:
                monitor_joints = [mid for mid in MONITOR_JOINT_IDS if mid in target_joints]
                q_text = " ".join(
                    [f"q{mid}={q_urdf[mid]:+.3f}rad/{np.degrees(q_urdf[mid]):+.1f}deg" for mid in monitor_joints]
                )
                torque_text = " ".join(
                    [f"tau{mid}={torque_motor[mid]:+.3f}Nm" for mid in monitor_joints]
                )
                thermal_text = ""
                if cfg["thermal_safety"].get("enabled", False):
                    thermal_parts = [
                        f"T{mid}={latest_thermal_temps[mid]:.1f}C"
                        for mid in cfg["thermal_safety"].get("monitor_motor_ids", [])
                        if mid in latest_thermal_temps
                    ]
                    thermal_text = " | " + (" ".join(thermal_parts) if thermal_parts else "temp=waiting")
                print(
                    f"cycle={cycle:05d} "
                    f"dt={elapsed*1000.0:.2f}ms "
                    f"alpha={ramp_alpha:.3f} | "
                    f"{q_text} | "
                    f"{torque_text}"
                    f"{thermal_text}"
                )

            cycle += 1

            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                overrun_count += 1

        total_time = time.monotonic() - loop_start_time
        achieved_rate = cycle / total_time if total_time > 1e-9 else 0.0

        print()
        print("Normal exit: ramping torque to zero, then sending Type4 disable to target joints")

        # Exit ramp to reduce the final torque drop.
        exit_ramp_sec = min(0.5, max(0.0, cfg["startup_ramp_sec"] * 0.25))
        if exit_ramp_sec > 0.0:
            exit_start = time.monotonic()
            exit_deadline = exit_start + exit_ramp_sec
            while time.monotonic() < exit_deadline:
                ratio = max(0.0, (exit_deadline - time.monotonic()) / exit_ramp_sec)
                exit_torque = {mid: prev_torque[mid] * ratio for mid in target_joints}

                ok, motor_pos_map, _reason = read_arm_positions_fast(driver, timeout=args.pos_timeout)
                if ok:
                    send_type1_targets(
                        driver=driver,
                        target_joints=target_joints,
                        motor_pos_map=motor_pos_map,
                        torque_motor=exit_torque,
                        zero_torque_kd=cfg["zero_torque_kd"],
                        delay=args.type1_delay,
                    )
                time.sleep(period)

        send_disable_targets(driver, target_joints)

        stats = driver.get_stats()

        print()
        print("Runtime stats:")
        print(f"  cycles:            {cycle}")
        print(f"  elapsed:           {total_time:.3f} s")
        print(f"  achieved_rate:     {achieved_rate:.2f} Hz")
        print(f"  overruns:          {overrun_count}")
        print(f"  read_fail_count:   {read_fail_count}")
        if cfg["software_damping"]["enabled"]:
            print("  max_abs_qdot_urdf:")
            for mid in [m for m in MONITOR_JOINT_IDS if m in target_joints]:
                print(f"    Joint{mid}: {max_abs_qdot_seen[mid]:.4f} rad/s")
            print("  max_abs_software_damping_torque:")
            for mid in [m for m in MONITOR_JOINT_IDS if m in target_joints]:
                print(f"    Joint{mid}: {max_abs_damping_seen[mid]:.4f} Nm")
        if cfg["joint2_anti_j3_coupling_hold"]["enabled"] and 2 in target_joints:
            print(f"  max_abs_joint2_anti_j3_hold_torque: {max_abs_joint2_anti_j3_hold_seen:.4f} Nm")

        if cycle_times:
            print(f"  cycle_dt_avg:      {statistics.mean(cycle_times)*1000.0:.2f} ms")
            print(f"  cycle_dt_median:   {statistics.median(cycle_times)*1000.0:.2f} ms")
            print(f"  cycle_dt_p90:      {percentile(cycle_times, 90)*1000.0:.2f} ms")
            print(f"  cycle_dt_p99:      {percentile(cycle_times, 99)*1000.0:.2f} ms")
            print(f"  cycle_dt_max:      {max(cycle_times)*1000.0:.2f} ms")

        print()
        print("Driver stats:")
        print(f"  rx_count:            {stats['rx_count']}")
        print(f"  tx_count:            {stats['tx_count']}")
        print(f"  rx_error_count:      {stats['rx_error_count']}")
        print(f"  tx_error_count:      {stats['tx_error_count']}")
        print(f"  unknown_frame_count: {stats['unknown_frame_count']}")
        print(f"  type2_feedback_count:{stats['type2_feedback_count']}")
        print(f"  param_reply_count:   {stats['param_reply_count']}")
        print()
        print("RESULT: PASS")
        print("SocketCAN zero-torque gravity mode completed.")

    except KeyboardInterrupt:
        print()
        if did_send_real_command:
            print("KeyboardInterrupt: sending Type4 disable to target joints")
            try:
                send_disable_targets(driver, target_joints)
            except Exception as e:
                print(f"Disable after KeyboardInterrupt failed: {e}")
        else:
            print("KeyboardInterrupt before armed command. No Type4 disable was sent.")
        print("RESULT: INTERRUPTED")
        return 130

    except Exception as e:
        print()
        print(f"ERROR: {e}")
        if did_send_real_command:
            print("Exception path: sending Type4 disable to target joints")
            try:
                send_disable_targets(driver, target_joints)
            except Exception as disable_error:
                print(f"Disable after exception failed: {disable_error}")
        else:
            print("Exception happened before armed command. No Type4 disable was sent.")
        print("RESULT: FAIL")
        return 1

    finally:
        if did_send_real_command:
            try:
                send_disable_targets(driver, target_joints)
            except Exception:
                pass
        driver.close()
        if joint_state_publisher is not None:
            joint_state_publisher.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())