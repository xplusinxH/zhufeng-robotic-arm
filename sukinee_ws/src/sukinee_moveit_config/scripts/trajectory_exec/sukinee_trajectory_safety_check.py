import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


DEFAULT_LIMITS_YAML = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "sukinee_trajectory_safety_limits.yaml"
)

DEFAULT_URDF = Path("/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf")
DEFAULT_MAIN_OFFSET_JSON = Path("/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json")
DEFAULT_GRAVITY_CONFIG_JSON = Path("/home/zzj/sukinee_ws/sukinee_gravity_assist_config.json")

# RobStride private protocol Type1 ranges currently used by sukinee_socketcan_driver.py.
TYPE1_LIMITS = {
    "RS00": {
        "p_min": -12.57,
        "p_max": 12.57,
        "v_min": -33.0,
        "v_max": 33.0,
        "t_min": -14.0,
        "t_max": 14.0,
        "kp_min": 0.0,
        "kp_max": 500.0,
        "kd_min": 0.0,
        "kd_max": 5.0,
    },
    "RS05": {
        "p_min": -12.57,
        "p_max": 12.57,
        "v_min": -50.0,
        "v_max": 50.0,
        "t_min": -5.5,
        "t_max": 5.5,
        "kp_min": 0.0,
        "kp_max": 500.0,
        "kd_min": 0.0,
        "kd_max": 5.0,
    },
}

MOTOR_TYPE_BY_JOINT = {
    "Joint1": "RS00",
    "Joint2": "RS00",
    "Joint3": "RS00",
    "Joint4": "RS05",
    "Joint5": "RS05",
    "Joint6": "RS05",
    "Joint7": "RS05",
}

MOTOR_ID_BY_JOINT = {
    "Joint1": 1,
    "Joint2": 2,
    "Joint3": 3,
    "Joint4": 4,
    "Joint5": 5,
    "Joint6": 6,
    "Joint7": 7,
}


class CheckReport:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.passes: List[str] = []

    def pass_(self, text: str) -> None:
        self.passes.append(text)
        print(f"[PASS] {text}")

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        print(f"[WARN] {text}")

    def fail(self, text: str) -> None:
        self.failures.append(text)
        print(f"[FAIL] {text}")

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "passes": list(self.passes),
            "warnings": list(self.warnings),
            "failures": list(self.failures),
        }


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dictionary: {path}")

    return data


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a dictionary: {path}")

    return data


def require_dict(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a dictionary.")
    return value


def require_list_of_str(data: Dict[str, Any], key: str) -> List[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"'{key}' must be a list of strings.")
    return list(value)


def require_float(data: Dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number.")
    return float(value)


def parse_position_limits(
    raw_limits: Dict[str, Any],
    expected_joints: List[str],
) -> Dict[str, Optional[Tuple[float, float]]]:
    limits: Dict[str, Optional[Tuple[float, float]]] = {}

    for joint in expected_joints:
        raw = raw_limits.get(joint)

        if raw is None:
            limits[joint] = None
            continue

        if not isinstance(raw, dict):
            raise ValueError(
                f"position_limits_rad.{joint} must be null or a dict with lower/upper."
            )

        if "lower" not in raw or "upper" not in raw:
            raise ValueError(f"position_limits_rad.{joint} must contain lower and upper.")

        lower = float(raw["lower"])
        upper = float(raw["upper"])

        if lower >= upper:
            raise ValueError(
                f"Invalid position limit for {joint}: lower {lower} >= upper {upper}."
            )

        limits[joint] = (lower, upper)

    return limits


def parse_per_joint_float_map(
    raw: Dict[str, Any],
    expected_joints: List[str],
    key_name: str,
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    for joint in expected_joints:
        if joint not in raw:
            raise ValueError(f"Missing {key_name}.{joint}")
        value = raw[joint]
        if not isinstance(value, (int, float)):
            raise ValueError(f"{key_name}.{joint} must be a number.")
        result[joint] = float(value)

    return result


def load_limits_config(path: Path) -> Dict[str, Any]:
    cfg = load_yaml(path)

    expected_joints = require_list_of_str(cfg, "expected_joints")
    forbidden_joints = require_list_of_str(cfg, "forbidden_joints")

    position_limits_raw = require_dict(cfg, "position_limits_rad")
    position_limits = parse_position_limits(position_limits_raw, expected_joints)

    max_joint_step_rad = require_float(cfg, "max_joint_step_rad")
    min_duration_sec = require_float(cfg, "min_duration_sec")

    velocity_raw = require_dict(cfg, "max_abs_velocity_rad_s")
    acceleration_raw = require_dict(cfg, "max_abs_acceleration_rad_s2")

    max_abs_velocity = parse_per_joint_float_map(
        velocity_raw, expected_joints, "max_abs_velocity_rad_s"
    )
    max_abs_acceleration = parse_per_joint_float_map(
        acceleration_raw, expected_joints, "max_abs_acceleration_rad_s2"
    )

    if max_joint_step_rad <= 0:
        raise ValueError("max_joint_step_rad must be positive.")
    if min_duration_sec < 0:
        raise ValueError("min_duration_sec must be non-negative.")

    return {
        "expected_joints": expected_joints,
        "forbidden_joints": forbidden_joints,
        "position_limits": position_limits,
        "max_joint_step_rad": max_joint_step_rad,
        "min_duration_sec": min_duration_sec,
        "max_abs_velocity": max_abs_velocity,
        "max_abs_acceleration": max_abs_acceleration,
    }


def get_trajectory_block(data: Dict[str, Any]) -> Dict[str, Any]:
    traj = data.get("trajectory")
    if not isinstance(traj, dict):
        raise ValueError("Trajectory YAML must contain a 'trajectory' dictionary.")
    return traj


def validate_joint_names(
    traj: Dict[str, Any],
    expected_joints: List[str],
    forbidden_joints: List[str],
    report: CheckReport,
) -> List[str]:
    joint_names = traj.get("joint_names")

    if not isinstance(joint_names, list):
        report.fail("'trajectory.joint_names' is missing or not a list.")
        return []

    joint_names = [str(j) for j in joint_names]

    for joint in forbidden_joints:
        if joint in joint_names:
            report.fail(f"{joint} appears in arm trajectory. It must be excluded.")
        else:
            report.pass_(f"{joint} is not included in arm trajectory.")

    if joint_names != expected_joints:
        report.fail(f"joint_names must be exactly {expected_joints}, got {joint_names}.")
    else:
        report.pass_(f"joint_names exactly match expected arm joints: {expected_joints}")

    return joint_names


def validate_points_basic(
    traj: Dict[str, Any],
    joint_names: List[str],
    report: CheckReport,
) -> List[Dict[str, Any]]:
    points = traj.get("points")

    if not isinstance(points, list) or not points:
        report.fail("'trajectory.points' is missing or empty.")
        return []

    n = len(joint_names)

    for i, p in enumerate(points):
        if not isinstance(p, dict):
            report.fail(f"Point {i} is not a dictionary.")
            continue

        if "time_from_start_sec" not in p:
            report.fail(f"Point {i} missing time_from_start_sec.")

        positions = p.get("positions")
        if not isinstance(positions, list):
            report.fail(f"Point {i} missing positions list.")
        elif len(positions) != n:
            report.fail(f"Point {i} positions length {len(positions)} != {n}.")

        for field in ["velocities", "accelerations"]:
            values = p.get(field)
            if values is not None:
                if not isinstance(values, list):
                    report.fail(f"Point {i} field '{field}' exists but is not a list.")
                elif len(values) != n:
                    report.fail(f"Point {i} {field} length {len(values)} != {n}.")

    if report.ok:
        report.pass_(f"All {len(points)} trajectory points have valid basic structure.")

    return points


def extract_times_positions(
    points: List[Dict[str, Any]],
    joint_count: int,
) -> Tuple[List[float], List[List[float]]]:
    times: List[float] = []
    positions: List[List[float]] = []

    for i, p in enumerate(points):
        t = float(p["time_from_start_sec"])
        q = [float(x) for x in p["positions"]]

        if len(q) != joint_count:
            raise ValueError(f"Point {i} positions length mismatch.")

        times.append(t)
        positions.append(q)

    return times, positions


def validate_time(
    times: List[float],
    min_duration_sec: float,
    report: CheckReport,
) -> None:
    if not times:
        report.fail("No times found.")
        return

    if abs(times[0]) > 1e-9:
        report.warn(f"First time_from_start_sec is {times[0]:.9f}, not exactly 0.0.")
    else:
        report.pass_("First time_from_start_sec is 0.0.")

    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            report.fail(
                f"time_from_start_sec must be strictly increasing: "
                f"point {i-1}={times[i-1]:.9f}, point {i}={times[i]:.9f}"
            )
            return

    report.pass_("time_from_start_sec is strictly increasing.")

    duration = times[-1] - times[0]
    if duration < min_duration_sec:
        report.fail(
            f"Trajectory duration {duration:.6f} sec is shorter than "
            f"min_duration_sec {min_duration_sec:.6f}."
        )
    else:
        report.pass_(
            f"Trajectory duration {duration:.6f} sec >= min_duration_sec {min_duration_sec:.6f}."
        )


def validate_position_limits(
    joint_names: List[str],
    positions: List[List[float]],
    position_limits: Dict[str, Optional[Tuple[float, float]]],
    report: CheckReport,
    eps: float = 1e-6,
) -> None:
    for j_idx, joint in enumerate(joint_names):
        values = [q[j_idx] for q in positions]
        min_v = min(values)
        max_v = max(values)

        limit = position_limits.get(joint)
        if limit is None:
            report.warn(
                f"{joint} has no lower/upper position limit in safety config; "
                f"position check skipped. observed range=[{min_v:.6f}, {max_v:.6f}]"
            )
            continue

        lower, upper = limit
        if min_v < lower - eps or max_v > upper + eps:
            report.fail(
                f"{joint} position out of configured limit. "
                f"observed=[{min_v:.6f}, {max_v:.6f}], limit=[{lower:.6f}, {upper:.6f}]"
            )
        else:
            report.pass_(
                f"{joint} position within configured limit. "
                f"observed=[{min_v:.6f}, {max_v:.6f}], limit=[{lower:.6f}, {upper:.6f}]"
            )


def validate_max_joint_step(
    joint_names: List[str],
    positions: List[List[float]],
    max_joint_step_rad: float,
    report: CheckReport,
) -> None:
    if len(positions) < 2:
        report.fail("Need at least 2 points to check joint step.")
        return

    max_step_by_joint = {joint: 0.0 for joint in joint_names}
    max_step_info = {joint: -1 for joint in joint_names}

    for i in range(1, len(positions)):
        prev = positions[i - 1]
        cur = positions[i]
        for j_idx, joint in enumerate(joint_names):
            step = abs(cur[j_idx] - prev[j_idx])
            if step > max_step_by_joint[joint]:
                max_step_by_joint[joint] = step
                max_step_info[joint] = i

    any_fail = False
    for joint in joint_names:
        step = max_step_by_joint[joint]
        idx = max_step_info[joint]
        if step > max_joint_step_rad:
            any_fail = True
            report.fail(
                f"{joint} max adjacent step {step:.6f} rad at point {idx} "
                f"> threshold {max_joint_step_rad:.6f} rad."
            )
        else:
            report.pass_(
                f"{joint} max adjacent step {step:.6f} rad <= threshold {max_joint_step_rad:.6f} rad."
            )

    if not any_fail:
        report.pass_("All joint adjacent steps are within threshold.")


def get_vector_series_from_points(
    points: List[Dict[str, Any]],
    field: str,
    joint_count: int,
) -> Optional[List[List[float]]]:
    if any(field not in p for p in points):
        return None

    series: List[List[float]] = []
    for i, p in enumerate(points):
        values = p.get(field)
        if not isinstance(values, list) or len(values) != joint_count:
            raise ValueError(f"Point {i} invalid field '{field}'.")
        series.append([float(x) for x in values])

    return series


def estimate_velocities(
    times: List[float],
    positions: List[List[float]],
) -> List[List[float]]:
    n_points = len(positions)
    n_joints = len(positions[0])

    velocities = [[0.0 for _ in range(n_joints)] for _ in range(n_points)]

    if n_points < 2:
        return velocities

    for i in range(n_points):
        if i == 0:
            dt = times[1] - times[0]
            q0 = positions[0]
            q1 = positions[1]
        else:
            dt = times[i] - times[i - 1]
            q0 = positions[i - 1]
            q1 = positions[i]

        if dt <= 0:
            continue

        for j in range(n_joints):
            velocities[i][j] = (q1[j] - q0[j]) / dt

    return velocities


def estimate_accelerations(
    times: List[float],
    velocities: List[List[float]],
) -> List[List[float]]:
    n_points = len(velocities)
    n_joints = len(velocities[0])

    accelerations = [[0.0 for _ in range(n_joints)] for _ in range(n_points)]

    if n_points < 2:
        return accelerations

    for i in range(1, n_points):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        for j in range(n_joints):
            accelerations[i][j] = (velocities[i][j] - velocities[i - 1][j]) / dt

    return accelerations


def validate_threshold_series(
    joint_names: List[str],
    series: List[List[float]],
    thresholds: Dict[str, float],
    label: str,
    unit: str,
    report: CheckReport,
) -> Dict[str, float]:
    observed: Dict[str, float] = {}
    for j_idx, joint in enumerate(joint_names):
        values = [row[j_idx] for row in series]
        max_abs = max(abs(v) for v in values)
        threshold = thresholds[joint]
        observed[joint] = max_abs

        if max_abs > threshold + 1e-9:
            report.fail(
                f"{joint} max_abs_{label} {max_abs:.6f} {unit} "
                f"> threshold {threshold:.6f} {unit}."
            )
        else:
            report.pass_(
                f"{joint} max_abs_{label} {max_abs:.6f} {unit} "
                f"<= threshold {threshold:.6f} {unit}."
            )
    return observed


def parse_joint_float_map(text: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not text:
        return result
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected JointN:value item, got '{item}'.")
        key, value = item.split(":", 1)
        key = key.strip()
        if key.startswith("joint"):
            key = "Joint" + key[5:]
        elif key.startswith("J") and key[1:].isdigit():
            key = "Joint" + key[1:]
        result[key] = float(value.strip())
    return result


def load_motor_offsets(path: Path, expected_joints: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    data = load_json(path)
    signs_raw = data.get("motor_to_urdf_sign")
    zeros_raw = data.get("motor_pos_at_urdf_zero")
    if not isinstance(signs_raw, dict) or not isinstance(zeros_raw, dict):
        raise ValueError(
            "Offset JSON must contain 'motor_to_urdf_sign' and 'motor_pos_at_urdf_zero'."
        )

    signs: Dict[str, float] = {}
    zeros: Dict[str, float] = {}
    for joint in expected_joints:
        if joint not in signs_raw or joint not in zeros_raw:
            raise ValueError(f"Offset JSON missing {joint} sign or zero.")
        signs[joint] = float(signs_raw[joint])
        zeros[joint] = float(zeros_raw[joint])
        if signs[joint] not in (-1.0, 1.0):
            raise ValueError(f"Offset sign for {joint} must be +1 or -1, got {signs[joint]}")
    return signs, zeros


def load_gravity_config(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if "gravity_feedforward_ratio" not in data:
        raise ValueError("Gravity config missing gravity_feedforward_ratio.")
    if "gravity_joint_scale" not in data or not isinstance(data["gravity_joint_scale"], dict):
        raise ValueError("Gravity config missing gravity_joint_scale dict.")
    if "max_abs_torque" not in data or not isinstance(data["max_abs_torque"], dict):
        raise ValueError("Gravity config missing max_abs_torque dict.")
    return data


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def effective_joint_scale(joint: str, q: float, cfg: Dict[str, Any]) -> float:
    motor_id = MOTOR_ID_BY_JOINT[joint]
    scale_raw = cfg.get("gravity_joint_scale", {})
    base_scale = float(scale_raw.get(str(motor_id), scale_raw.get(joint, 1.0)))

    if joint == "Joint4":
        angle_cfg = cfg.get("joint4_angle_dependent_gravity_scale", {})
        if isinstance(angle_cfg, dict) and bool(angle_cfg.get("enabled", False)):
            q_start = float(angle_cfg.get("q_start_rad", 0.45))
            q_full = float(angle_cfg.get("q_full_rad", 0.80))
            if q_full > q_start:
                scale_start = float(angle_cfg.get("scale_at_start", base_scale))
                scale_full = float(angle_cfg.get("scale_at_full", base_scale))
                t = max(0.0, min(1.0, (q - q_start) / (q_full - q_start)))
                if str(angle_cfg.get("blend", "smoothstep")) == "smoothstep":
                    t = smoothstep01(t)
                return scale_start + t * (scale_full - scale_start)
    return base_scale


def max_abs_torque_limit(joint: str, cfg: Dict[str, Any]) -> Optional[float]:
    motor_id = MOTOR_ID_BY_JOINT[joint]
    raw = cfg.get("max_abs_torque", {})
    if str(motor_id) in raw:
        return float(raw[str(motor_id)])
    if joint in raw:
        return float(raw[joint])
    return None


def check_value_range(
    value: float,
    lower: float,
    upper: float,
    label: str,
    report: CheckReport,
    eps: float = 1e-9,
) -> bool:
    if value < lower - eps or value > upper + eps:
        report.fail(f"{label} {value:.6f} outside [{lower:.6f}, {upper:.6f}].")
        return False
    return True


def build_pinocchio_model(urdf_path: Path):
    try:
        import pinocchio as pin  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Pinocchio is required for gravity precheck but could not be imported. "
            f"Import error: {exc}"
        )

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return pin, model, data


def compute_pinocchio_gravity_series(
    urdf_path: Path,
    joint_names: List[str],
    positions: List[List[float]],
) -> List[Dict[str, float]]:
    pin, model, data = build_pinocchio_model(urdf_path)

    joint_index: Dict[str, Tuple[int, int]] = {}
    for joint in joint_names:
        jid = model.getJointId(joint)
        if jid >= len(model.joints):
            raise ValueError(f"Joint '{joint}' not found in Pinocchio model.")
        idx_q = model.joints[jid].idx_q
        idx_v = model.joints[jid].idx_v
        joint_index[joint] = (idx_q, idx_v)

    series: List[Dict[str, float]] = []
    for q_row in positions:
        q_model = pin.neutral(model)
        for j_idx, joint in enumerate(joint_names):
            idx_q, _idx_v = joint_index[joint]
            q_model[idx_q] = float(q_row[j_idx])

        # Same zero-velocity inverse dynamics interpretation used in zero_drag.
        # Pinocchio's Python binding requires Eigen-compatible numpy arrays here;
        # plain Python lists can fail on some ROS/Jazzy Pinocchio builds.
        import numpy as np

        v_model = np.zeros(model.nv, dtype=float)
        a_model = np.zeros(model.nv, dtype=float)
        tau_vec = pin.rnea(model, data, q_model, v_model, a_model)

        tau_map: Dict[str, float] = {}
        for joint in joint_names:
            _idx_q, idx_v = joint_index[joint]
            tau_map[joint] = float(tau_vec[idx_v])
        series.append(tau_map)

    return series


def run_execution_precheck(
    joint_names: List[str],
    times: List[float],
    positions: List[List[float]],
    velocities: List[List[float]],
    args: argparse.Namespace,
    report: CheckReport,
) -> Dict[str, Any]:
    print("")
    print("Sukinee Type1 execution precheck")
    print("  boundary: offline only; no CAN socket; no Type1/Type3/Type4/Type6; no MoveIt real execution")

    offset_json = Path(args.offset_json).expanduser().resolve()
    config_json = Path(args.config_json).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()

    signs, zeros = load_motor_offsets(offset_json, joint_names)
    cfg = load_gravity_config(config_json)

    print(f"  offset_json: {offset_json}")
    print(f"  config_json: {config_json}")
    print(f"  urdf: {urdf}")
    print(f"  preview_kp: {args.preview_kp}")
    print(f"  preview_kd: {args.preview_kd}")

    if args.preview_kp < 0 or args.preview_kd < 0:
        report.fail("preview_kp and preview_kd must be non-negative.")
    if args.preview_kp > 500.0:
        report.fail("preview_kp exceeds Type1 maximum 500.0.")
    if args.preview_kd > 5.0:
        report.fail("preview_kd exceeds Type1 maximum 5.0.")

    current_q = parse_joint_float_map(args.current_q) if args.current_q else {}
    if current_q:
        first = positions[0]
        max_start_error = 0.0
        for j_idx, joint in enumerate(joint_names):
            if joint not in current_q:
                report.fail(f"--current-q missing {joint}.")
                continue
            err = abs(first[j_idx] - current_q[joint])
            max_start_error = max(max_start_error, err)
            if err > args.max_start_error_rad:
                report.fail(
                    f"trajectory start mismatch for {joint}: first waypoint {first[j_idx]:.6f}, "
                    f"current {current_q[joint]:.6f}, error {err:.6f} > {args.max_start_error_rad:.6f} rad."
                )
        if max_start_error <= args.max_start_error_rad:
            report.pass_(
                f"trajectory start is close to --current-q; max error {max_start_error:.6f} rad "
                f"<= {args.max_start_error_rad:.6f} rad."
            )
    else:
        report.warn("No --current-q provided; start-to-current consistency check skipped.")

    # Gravity estimation is optional but enabled by default inside execution-precheck.
    gravity_series: Optional[List[Dict[str, float]]] = None
    if args.skip_gravity_precheck:
        report.warn("Gravity precheck skipped by --skip-gravity-precheck.")
    else:
        gravity_series = compute_pinocchio_gravity_series(urdf, joint_names, positions)
        report.pass_("Pinocchio gravity series computed from URDF.")

    # Summary accumulators.
    summary: Dict[str, Any] = {
        "execution_precheck_enabled": True,
        "safety_boundary": {
            "socketcan": False,
            "type1_sent": False,
            "type3_enable": False,
            "type4_disable": False,
            "type6_set_zero": False,
            "save_motor_parameters": False,
            "change_can_id": False,
            "switch_protocol": False,
            "moveit_real_execution": False,
        },
        "offset_json": str(offset_json),
        "config_json": str(config_json),
        "urdf": str(urdf),
        "preview_kp": float(args.preview_kp),
        "preview_kd": float(args.preview_kd),
        "per_joint": {},
    }

    ratio = float(cfg.get("gravity_feedforward_ratio", 1.0))

    for j_idx, joint in enumerate(joint_names):
        motor_type = MOTOR_TYPE_BY_JOINT[joint]
        limits = TYPE1_LIMITS[motor_type]
        sign = signs[joint]
        zero = zeros[joint]
        motor_id = MOTOR_ID_BY_JOINT[joint]

        motor_positions = []
        motor_velocities = []
        motor_gravity_ff = []
        effective_scales = []

        for p_idx, q_row in enumerate(positions):
            q = float(q_row[j_idx])
            qdot = float(velocities[p_idx][j_idx])
            motor_positions.append(zero + sign * q)
            motor_velocities.append(sign * qdot)

            if gravity_series is not None:
                scale = effective_joint_scale(joint, q, cfg)
                tau_urdf = gravity_series[p_idx][joint]
                motor_ff = sign * tau_urdf * ratio * scale
                motor_gravity_ff.append(motor_ff)
                effective_scales.append(scale)

        min_pos = min(motor_positions)
        max_pos = max(motor_positions)
        max_abs_vel = max(abs(v) for v in motor_velocities)
        max_abs_ff = max(abs(v) for v in motor_gravity_ff) if motor_gravity_ff else 0.0
        max_abs_type1_torque = max_abs_ff
        cfg_limit = max_abs_torque_limit(joint, cfg)

        failure_count_before = len(report.failures)
        check_value_range(min_pos, limits["p_min"], limits["p_max"], f"{joint}/motor{motor_id} min Type1 position", report)
        check_value_range(max_pos, limits["p_min"], limits["p_max"], f"{joint}/motor{motor_id} max Type1 position", report)
        check_value_range(-max_abs_vel, limits["v_min"], limits["v_max"], f"{joint}/motor{motor_id} negative Type1 velocity bound", report)
        check_value_range(max_abs_vel, limits["v_min"], limits["v_max"], f"{joint}/motor{motor_id} positive Type1 velocity bound", report)
        check_value_range(args.preview_kp, limits["kp_min"], limits["kp_max"], f"{joint}/motor{motor_id} preview_kp", report)
        check_value_range(args.preview_kd, limits["kd_min"], limits["kd_max"], f"{joint}/motor{motor_id} preview_kd", report)

        if gravity_series is not None:
            check_value_range(-max_abs_ff, limits["t_min"], limits["t_max"], f"{joint}/motor{motor_id} negative Type1 torque_ff bound", report)
            check_value_range(max_abs_ff, limits["t_min"], limits["t_max"], f"{joint}/motor{motor_id} positive Type1 torque_ff bound", report)
            if cfg_limit is None:
                report.warn(f"{joint}/motor{motor_id} has no max_abs_torque entry in gravity config.")
            elif max_abs_ff > cfg_limit + 1e-9:
                report.fail(
                    f"{joint}/motor{motor_id} max_abs gravity torque_ff {max_abs_ff:.6f} Nm "
                    f"> config max_abs_torque {cfg_limit:.6f} Nm."
                )
            else:
                report.pass_(
                    f"{joint}/motor{motor_id} max_abs gravity torque_ff {max_abs_ff:.6f} Nm "
                    f"<= config max_abs_torque {cfg_limit:.6f} Nm."
                )

        if len(report.failures) == failure_count_before:
            # Print one compact PASS when no new failure was created for this joint.
            report.pass_(
                f"{joint}/motor{motor_id} Type1 preview in range: "
                f"pos=[{min_pos:.6f}, {max_pos:.6f}], max|vel|={max_abs_vel:.6f}, "
                f"max|gravity_ff|={max_abs_ff:.6f} Nm."
            )

        summary["per_joint"][joint] = {
            "motor_id": motor_id,
            "motor_type": motor_type,
            "sign": sign,
            "zero": zero,
            "type1_limits": limits,
            "motor_position_range": [min_pos, max_pos],
            "max_abs_motor_velocity": max_abs_vel,
            "max_abs_gravity_torque_ff": max_abs_ff,
            "config_max_abs_torque": cfg_limit,
            "effective_scale_range": [min(effective_scales), max(effective_scales)] if effective_scales else None,
        }

    return summary


def observed_trajectory_summary(
    joint_names: List[str],
    times: List[float],
    positions: List[List[float]],
    velocities: List[List[float]],
    accelerations: List[List[float]],
) -> Dict[str, Any]:
    per_joint: Dict[str, Any] = {}
    for j_idx, joint in enumerate(joint_names):
        q_vals = [row[j_idx] for row in positions]
        v_vals = [row[j_idx] for row in velocities]
        a_vals = [row[j_idx] for row in accelerations]
        per_joint[joint] = {
            "q_min": min(q_vals),
            "q_max": max(q_vals),
            "max_abs_velocity": max(abs(v) for v in v_vals),
            "max_abs_acceleration": max(abs(a) for a in a_vals),
        }
    return {
        "point_count": len(times),
        "duration_sec": times[-1] - times[0] if times else 0.0,
        "per_joint": per_joint,
    }


def print_summary(
    joint_names: List[str],
    times: List[float],
    positions: List[List[float]],
    velocities: List[List[float]],
    accelerations: List[List[float]],
) -> None:
    print("")
    print("Observed trajectory summary:")
    print(f"  point_count: {len(times)}")
    print(f"  duration_sec: {times[-1] - times[0]:.6f}")

    for j_idx, joint in enumerate(joint_names):
        q_vals = [row[j_idx] for row in positions]
        v_vals = [row[j_idx] for row in velocities]
        a_vals = [row[j_idx] for row in accelerations]

        print(
            f"  {joint}: "
            f"q=[{min(q_vals): .6f}, {max(q_vals): .6f}] rad, "
            f"max|v|={max(abs(v) for v in v_vals): .6f} rad/s, "
            f"max|a|={max(abs(a) for a in a_vals): .6f} rad/s^2"
        )


def print_loaded_limits(limits: Dict[str, Any], limits_yaml: Path) -> None:
    expected_joints = limits["expected_joints"]
    max_abs_velocity = limits["max_abs_velocity"]
    max_abs_acceleration = limits["max_abs_acceleration"]

    print(f"  limits_yaml: {limits_yaml}")
    print(f"  expected_joints: {expected_joints}")
    print(f"  max_joint_step_rad: {limits['max_joint_step_rad']}")
    print(f"  min_duration_sec: {limits['min_duration_sec']}")

    print("  max_abs_velocity_rad_s:")
    for joint in expected_joints:
        print(f"    {joint}: {max_abs_velocity[joint]}")

    print("  max_abs_acceleration_rad_s2:")
    for joint in expected_joints:
        print(f"    {joint}: {max_abs_acceleration[joint]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline safety check for Sukinee MoveIt trajectory YAML, with optional Type1 execution precheck."
    )

    parser.add_argument(
        "--trajectory-yaml",
        required=True,
        help="Path to trajectory YAML exported by sukinee_capture_display_trajectory.py",
    )

    parser.add_argument(
        "--limits-yaml",
        default=str(DEFAULT_LIMITS_YAML),
        help=(
            "Path to safety limits YAML. Default: "
            "../../config/sukinee_trajectory_safety_limits.yaml relative to this script."
        ),
    )

    parser.add_argument(
        "--execution-precheck",
        action="store_true",
        help="Also preview Sukinee Type1 motor mapping / gravity feedforward limits. Sends no CAN commands.",
    )

    parser.add_argument(
        "--offset-json",
        default=str(DEFAULT_MAIN_OFFSET_JSON),
        help="Offset JSON for q_urdf <-> motor_pos mapping. Used only with --execution-precheck.",
    )

    parser.add_argument(
        "--config-json",
        default=str(DEFAULT_GRAVITY_CONFIG_JSON),
        help="Gravity assist config JSON. Used only with --execution-precheck.",
    )

    parser.add_argument(
        "--urdf",
        default=str(DEFAULT_URDF),
        help="URDF used for Pinocchio gravity precheck. Used only with --execution-precheck.",
    )

    parser.add_argument(
        "--current-q",
        default="",
        help=(
            "Optional current q_urdf for trajectory start check, e.g. "
            "Joint1:0,Joint2:-0.08,Joint3:0.09,Joint4:0.05,Joint5:0,Joint6:0"
        ),
    )

    parser.add_argument(
        "--max-start-error-rad",
        type=float,
        default=0.05,
        help="Maximum allowed abs difference between --current-q and first trajectory waypoint.",
    )

    parser.add_argument(
        "--preview-kp",
        type=float,
        default=0.0,
        help="Preview Kp for future Type1 execution range check only. This script does not send it.",
    )

    parser.add_argument(
        "--preview-kd",
        type=float,
        default=0.0,
        help="Preview Kd for future Type1 execution range check only. This script does not send it.",
    )

    parser.add_argument(
        "--skip-gravity-precheck",
        action="store_true",
        help="Skip Pinocchio gravity and torque_ff precheck even when --execution-precheck is enabled.",
    )

    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write machine-readable safety summary JSON.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = CheckReport()
    summary: Dict[str, Any] = {
        "script": "sukinee_trajectory_safety_check.py",
        "mode": "offline_safety_check",
        "safety_boundary": {
            "socketcan": False,
            "type1_sent": False,
            "type3_enable": False,
            "type4_disable": False,
            "type6_set_zero": False,
            "save_motor_parameters": False,
            "change_can_id": False,
            "switch_protocol": False,
            "moveit_real_execution": False,
        },
    }

    trajectory_yaml = Path(args.trajectory_yaml).expanduser().resolve()
    limits_yaml = Path(args.limits_yaml).expanduser().resolve()

    print("Sukinee trajectory offline safety check")
    print(f"  trajectory_yaml: {trajectory_yaml}")
    print("  safety boundary: no SocketCAN, no real motor command, no MoveIt real execution")

    try:
        limits = load_limits_config(limits_yaml)
        print("")
        print("Loaded safety limits:")
        print_loaded_limits(limits, limits_yaml)
        print("")

        data = load_yaml(trajectory_yaml)
        traj = get_trajectory_block(data)

        expected_joints = limits["expected_joints"]
        forbidden_joints = limits["forbidden_joints"]
        position_limits = limits["position_limits"]

        joint_names = validate_joint_names(
            traj=traj,
            expected_joints=expected_joints,
            forbidden_joints=forbidden_joints,
            report=report,
        )

        if not joint_names:
            print("\nRESULT: FAIL")
            return 2

        points = validate_points_basic(traj, joint_names, report)
        if not points:
            print("\nRESULT: FAIL")
            return 2

        times, positions = extract_times_positions(points, len(joint_names))

        validate_time(times, limits["min_duration_sec"], report)
        validate_position_limits(joint_names, positions, position_limits, report)
        validate_max_joint_step(
            joint_names,
            positions,
            limits["max_joint_step_rad"],
            report,
        )

        velocities = get_vector_series_from_points(points, "velocities", len(joint_names))
        if velocities is None:
            report.warn("No complete velocities in YAML; estimating velocities from positions/time.")
            velocities = estimate_velocities(times, positions)
        else:
            report.pass_("Velocity vectors found in YAML.")

        accelerations = get_vector_series_from_points(points, "accelerations", len(joint_names))
        if accelerations is None:
            report.warn("No complete accelerations in YAML; estimating accelerations from velocity/time.")
            accelerations = estimate_accelerations(times, velocities)
        else:
            report.pass_("Acceleration vectors found in YAML.")

        observed_velocity = validate_threshold_series(
            joint_names=joint_names,
            series=velocities,
            thresholds=limits["max_abs_velocity"],
            label="velocity",
            unit="rad/s",
            report=report,
        )

        observed_acceleration = validate_threshold_series(
            joint_names=joint_names,
            series=accelerations,
            thresholds=limits["max_abs_acceleration"],
            label="acceleration",
            unit="rad/s^2",
            report=report,
        )

        print_summary(joint_names, times, positions, velocities, accelerations)

        summary["trajectory_yaml"] = str(trajectory_yaml)
        summary["limits_yaml"] = str(limits_yaml)
        summary["trajectory"] = observed_trajectory_summary(
            joint_names, times, positions, velocities, accelerations
        )
        summary["observed_max_abs_velocity_rad_s"] = observed_velocity
        summary["observed_max_abs_acceleration_rad_s2"] = observed_acceleration

        if args.execution_precheck:
            summary["execution_precheck"] = run_execution_precheck(
                joint_names=joint_names,
                times=times,
                positions=positions,
                velocities=velocities,
                args=args,
                report=report,
            )
        else:
            report.warn("--execution-precheck not enabled; Sukinee Type1/motor/gravity preview skipped.")
            summary["execution_precheck"] = {"enabled": False}

    except Exception as exc:
        report.fail(f"Exception while checking trajectory: {exc}")

    summary["report"] = report.as_dict()

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWrote summary JSON: {out}")

    print("")
    if report.ok:
        print("RESULT: PASS")
        if report.warnings:
            print(f"Warnings: {len(report.warnings)}")
        print("No CAN command was sent. This is not authorization for real automatic execution.")
        return 0

    print("RESULT: FAIL")
    print(f"Failures: {len(report.failures)}")
    print("No CAN command was sent.")
    return 2


if __name__ == "__main__":
    sys.exit(main())