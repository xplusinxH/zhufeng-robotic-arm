#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[2]
DRIVER_DIR = SRC_DIR / "sukinee_urdf" / "scripts" / "zero_drag"
if not DRIVER_DIR.exists():
    DRIVER_DIR = Path("/home/zzj/sukinee_ws/src/sukinee_urdf/scripts/zero_drag")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import sukinee_trajectory_safety_check as safety  # noqa: E402
from sukinee_socketcan_driver import SukineeSocketCANDriver  # noqa: E402


VERSION = "v1_3_publish_passive_gripper_joints"
CONFIRM_TEXT = "I_UNDERSTAND_THIS_SENDS_REAL_TYPE1_TRAJECTORY_COMMANDS"
POS_INDEX = 0x7019
ARM_MOTOR_IDS = [1, 2, 3, 4, 5, 6]

DEFAULT_OFFSET_JSON = (
    "/home/zzj/sukinee_ws/vision_calibration/data/run_100_20260619_012357/"
    "gate5_result_77mm_exclude_P0021_P0008_joint2prior001/"
    "sukinee_motor_to_urdf_offsets_calibrated.json"
)
DEFAULT_CONFIG_JSON = "/home/zzj/sukinee_ws/sukinee_gravity_assist_config.json"
DEFAULT_URDF = "/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf"
DEFAULT_OUT = "/home/zzj/sukinee_ws/trajectory_exec_reports/type1_executor_v1_report.json"


def load_and_check_trajectory(args: argparse.Namespace):
    report = safety.CheckReport()

    trajectory_yaml = Path(args.trajectory_yaml).expanduser().resolve()
    limits_yaml = Path(args.limits_yaml).expanduser().resolve()

    print("Sukinee Type1 trajectory executor v1")
    print(f"  version: {VERSION}")
    print(f"  mode: {'ARMED REAL TYPE1 EXECUTION' if args.armed else 'DRY-RUN'}")
    print("  trajectory_yaml:", trajectory_yaml)
    print("  limits_yaml:    ", limits_yaml)
    print("  offset_json:    ", Path(args.offset_json).expanduser().resolve())
    print("  config_json:    ", Path(args.config_json).expanduser().resolve())
    print("  urdf:           ", Path(args.urdf).expanduser().resolve())
    print("  output_json:    ", Path(args.output_json).expanduser())
    print("  rate:           ", f"{args.rate:.3f} Hz")
    print("  kp:             ", f"{args.preview_kp:.6f}")
    print("  kd:             ", f"{args.preview_kd:.6f}")
    print()

    data = safety.load_yaml(trajectory_yaml)
    limits = safety.load_limits_config(limits_yaml)
    traj = safety.get_trajectory_block(data)

    print("Loaded safety limits:")
    safety.print_loaded_limits(limits, limits_yaml)
    print()

    expected_joints = limits["expected_joints"]
    forbidden_joints = limits["forbidden_joints"]

    joint_names = safety.validate_joint_names(traj, expected_joints, forbidden_joints, report)
    points = safety.validate_points_basic(traj, joint_names, report)

    if not report.ok or not joint_names or not points:
        return False, report, {
            "joint_names": joint_names,
            "points": points,
            "times": [],
            "positions": [],
            "velocities": [],
            "accelerations": [],
            "observed": {},
            "execution_precheck": {},
        }

    times, positions = safety.extract_times_positions(points, len(joint_names))

    safety.validate_time(times, limits["min_duration_sec"], report)
    safety.validate_position_limits(joint_names, positions, limits["position_limits"], report)
    safety.validate_max_joint_step(joint_names, positions, limits["max_joint_step_rad"], report)

    velocities = safety.get_vector_series_from_points(points, "velocities", len(joint_names))
    if velocities is None:
        report.warn("Velocity vectors missing in YAML; estimating from adjacent waypoints.")
        velocities = safety.estimate_velocities(times, positions)
    else:
        report.pass_("Velocity vectors found in YAML.")

    accelerations = safety.get_vector_series_from_points(points, "accelerations", len(joint_names))
    if accelerations is None:
        report.warn("Acceleration vectors missing in YAML; estimating from velocities.")
        accelerations = safety.estimate_accelerations(times, velocities)
    else:
        report.pass_("Acceleration vectors found in YAML.")

    safety.validate_threshold_series(
        joint_names,
        velocities,
        limits["max_abs_velocity"],
        "velocity",
        "rad/s",
        report,
    )

    safety.validate_threshold_series(
        joint_names,
        accelerations,
        limits["max_abs_acceleration"],
        "acceleration",
        "rad/s^2",
        report,
    )

    observed = safety.observed_trajectory_summary(
        joint_names,
        times,
        positions,
        velocities,
        accelerations,
    )

    safety.print_summary(joint_names, times, positions, velocities, accelerations)

    execution_precheck = safety.run_execution_precheck(
        joint_names=joint_names,
        times=times,
        positions=positions,
        velocities=velocities,
        args=args,
        report=report,
    )

    context = {
        "joint_names": joint_names,
        "points": points,
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "observed": observed,
        "execution_precheck": execution_precheck,
    }

    return report.ok, report, context


def compute_type1_commands(args: argparse.Namespace, context: Dict[str, Any]):
    joint_names = context["joint_names"]
    times = context["times"]
    positions = context["positions"]
    velocities = context["velocities"]

    offset_json = Path(args.offset_json).expanduser().resolve()
    config_json = Path(args.config_json).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()

    signs, zeros = safety.load_motor_offsets(offset_json, joint_names)
    cfg = safety.load_gravity_config(config_json)
    ratio = float(cfg.get("gravity_feedforward_ratio", 1.0))

    gravity_series = None
    if not args.skip_gravity_precheck:
        gravity_series = safety.compute_pinocchio_gravity_series(urdf, joint_names, positions)

    commands = []

    for p_idx, t in enumerate(times):
        row = {
            "point_index": p_idx,
            "time_from_start_sec": float(t),
            "motors": {},
        }

        for j_idx, joint in enumerate(joint_names):
            motor_id = safety.MOTOR_ID_BY_JOINT[joint]
            sign = signs[joint]
            zero = zeros[joint]
            q = float(positions[p_idx][j_idx])
            qdot = float(velocities[p_idx][j_idx])

            motor_pos = zero + sign * q
            motor_vel = sign * qdot

            torque_ff = 0.0
            if gravity_series is not None:
                tau_urdf = gravity_series[p_idx][joint]
                scale = safety.effective_joint_scale(joint, q, cfg)
                torque_ff = sign * tau_urdf * ratio * scale

            row["motors"][str(motor_id)] = {
                "joint": joint,
                "motor_id": motor_id,
                "position": motor_pos,
                "velocity": motor_vel,
                "kp": float(args.preview_kp),
                "kd": float(args.preview_kd),
                "torque_ff": torque_ff,
            }

        commands.append(row)

    ranges = {}
    for joint in joint_names:
        motor_id = safety.MOTOR_ID_BY_JOINT[joint]
        vals = [cmd["motors"][str(motor_id)] for cmd in commands]
        ranges[joint] = {
            "motor_id": motor_id,
            "motor_position": {
                "min": min(v["position"] for v in vals),
                "max": max(v["position"] for v in vals),
            },
            "motor_velocity": {
                "min": min(v["velocity"] for v in vals),
                "max": max(v["velocity"] for v in vals),
                "max_abs": max(abs(v["velocity"]) for v in vals),
            },
            "kp": float(args.preview_kp),
            "kd": float(args.preview_kd),
            "torque_ff": {
                "min": min(v["torque_ff"] for v in vals),
                "max": max(v["torque_ff"] for v in vals),
                "max_abs": max(abs(v["torque_ff"]) for v in vals),
            },
        }

    return commands, ranges


def apply_real_execution_guards(args, context, command_ranges, report):
    duration = float(context["times"][-1] - context["times"][0]) if context["times"] else 0.0
    positions = context["positions"]
    joint_names = context["joint_names"]

    guard = {
        "enabled": bool(args.armed or args.enforce_real_guards),
        "max_real_duration_sec": float(args.max_real_duration_sec),
        "max_real_joint_motion_rad": float(args.max_real_joint_motion_rad),
        "min_real_kp": float(args.min_real_kp),
        "max_real_kp": float(args.max_real_kp),
        "max_real_kd": float(args.max_real_kd),
        "checks": [],
    }

    if not guard["enabled"]:
        return guard

    def add(ok, label, value=None):
        item = {"ok": bool(ok), "label": label, "value": value}
        guard["checks"].append(item)
        if ok:
            report.pass_(f"real guard: {label}")
        else:
            report.fail(f"real guard failed: {label}")

    add(
        duration <= args.max_real_duration_sec,
        f"duration {duration:.6f} <= {args.max_real_duration_sec:.6f} sec",
        duration,
    )

    for j_idx, joint in enumerate(joint_names):
        q_vals = [row[j_idx] for row in positions]
        motion = max(q_vals) - min(q_vals)
        add(
            motion <= args.max_real_joint_motion_rad,
            f"{joint} motion range {motion:.6f} <= {args.max_real_joint_motion_rad:.6f} rad",
            motion,
        )

    add(args.preview_kp >= args.min_real_kp, f"Kp {args.preview_kp:.6f} >= {args.min_real_kp:.6f}", args.preview_kp)
    add(args.preview_kp <= args.max_real_kp, f"Kp {args.preview_kp:.6f} <= {args.max_real_kp:.6f}", args.preview_kp)
    add(args.preview_kd <= args.max_real_kd, f"Kd {args.preview_kd:.6f} <= {args.max_real_kd:.6f}", args.preview_kd)

    for joint, r in command_ranges.items():
        motor_id = int(r["motor_id"])
        cfg_limit = context["execution_precheck"].get("per_joint", {}).get(joint, {}).get("config_max_abs_torque")
        max_abs_torque = float(r["torque_ff"]["max_abs"])
        if cfg_limit is not None:
            add(
                max_abs_torque <= float(cfg_limit) + 1e-9,
                f"{joint}/motor{motor_id} torque_ff {max_abs_torque:.6f} <= config max_abs_torque {float(cfg_limit):.6f}",
                max_abs_torque,
            )

    return guard


def verify_motor_start(driver, commands, max_error_rad, read_timeout):
    first = commands[0]["motors"]
    details = {
        "ok": True,
        "max_abs_error_rad": 0.0,
        "per_motor": {},
    }

    for motor_id in ARM_MOTOR_IDS:
        expected = float(first[str(motor_id)]["position"])
        status, value = driver.read_param_float(motor_id, POS_INDEX, timeout=read_timeout)

        if status != "OK" or value is None:
            details["ok"] = False
            details["per_motor"][str(motor_id)] = {
                "status": status,
                "actual_motor_pos": None,
                "expected_motor_pos": expected,
                "abs_error_rad": None,
                "ok": False,
            }
            continue

        actual = float(value)
        err = abs(actual - expected)
        ok = err <= max_error_rad

        details["max_abs_error_rad"] = max(details["max_abs_error_rad"], err)
        if not ok:
            details["ok"] = False

        details["per_motor"][str(motor_id)] = {
            "status": status,
            "actual_motor_pos": actual,
            "expected_motor_pos": expected,
            "abs_error_rad": err,
            "ok": ok,
        }

    return bool(details["ok"]), details


def send_disable_all(driver, delay):
    for motor_id in ARM_MOTOR_IDS:
        driver.send_disable(motor_id, clear_fault=False)
        time.sleep(delay)


def send_set_motion_mode_all(driver, delay):
    for motor_id in ARM_MOTOR_IDS:
        driver.send_set_motion_mode(motor_id)
        time.sleep(delay)


def send_enable_all(driver, delay):
    for motor_id in ARM_MOTOR_IDS:
        driver.send_enable(motor_id)
        time.sleep(delay)



JOINT_STATE_PUBLISHER = None
JOINT_NAMES_FOR_PUBLISH = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]

# Visualization-only passive joints.
# These are published to /joint_states so robot_state_publisher can generate gripper TF.
# They are NOT controlled by Type1 and are NOT included in trajectory safety checks.
PASSIVE_JOINT_STATES_FOR_PUBLISH = {
    "Joint7": 0.0,
    "left_finger": 0.0,
    "right_finger": 0.0,
}


class ExecutorJointStatePublisher:
    """
    Publish /joint_states during armed execution.

    source:
      - commanded: publish the commanded trajectory state converted back to q_urdf.
      - type2: publish measured Type2 feedback only when all six motors have fresh feedback.
      - auto: use fresh Type2 feedback when all six motors are available; otherwise fallback to commanded state.

    This class does not send CAN frames. It only reads cached Type2 feedback from the existing driver RX thread.
    """

    def __init__(self, enabled, topic, source, offset_json, max_type2_age_sec):
        self.enabled = bool(enabled)
        self.topic = str(topic)
        self.source = str(source)
        self.max_type2_age_sec = float(max_type2_age_sec)
        self.publish_count = 0
        self.commanded_publish_count = 0
        self.type2_publish_count = 0
        self.type2_missing_count = 0
        self.last_source = None
        self.node = None
        self.pub = None
        self.rclpy = None
        self.JointState = None

        self.signs = {}
        self.zeros = {}

        if not self.enabled:
            return

        import rclpy
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.JointState = JointState

        if not rclpy.ok():
            rclpy.init(args=[])

        self.node = rclpy.create_node("sukinee_type1_executor_joint_state_publisher")
        self.pub = self.node.create_publisher(JointState, self.topic, 10)

        signs, zeros = safety.load_motor_offsets(Path(offset_json).expanduser().resolve(), JOINT_NAMES_FOR_PUBLISH)
        self.signs = signs
        self.zeros = zeros

        print()
        print("Executor /joint_states publisher enabled:")
        print(f"  topic: {self.topic}")
        print(f"  source: {self.source}")
        print(f"  max_type2_age_sec: {self.max_type2_age_sec:.3f}")
        print("  note: source=auto uses measured Type2 only if all six motors have fresh Type2 feedback; otherwise commanded.")
        print()

    def _commanded_state_from_cmd(self, cmd):
        names = []
        positions = []
        velocities = []
        efforts = []

        for joint in JOINT_NAMES_FOR_PUBLISH:
            motor_id = safety.MOTOR_ID_BY_JOINT[joint]
            c = cmd["motors"][str(motor_id)]

            motor_pos = float(c["position"])
            motor_vel = float(c["velocity"])
            motor_torque = float(c["torque_ff"])

            sign = float(self.signs[joint])
            zero = float(self.zeros[joint])

            q = sign * (motor_pos - zero)
            qdot = sign * motor_vel
            effort = sign * motor_torque

            names.append(joint)
            positions.append(q)
            velocities.append(qdot)
            efforts.append(effort)

        for passive_joint, passive_q in PASSIVE_JOINT_STATES_FOR_PUBLISH.items():
            names.append(passive_joint)
            positions.append(float(passive_q))
            velocities.append(0.0)
            efforts.append(0.0)

        return names, positions, velocities, efforts

    def _type2_state_from_driver(self, driver):
        fbs = {}

        for joint in JOINT_NAMES_FOR_PUBLISH:
            motor_id = safety.MOTOR_ID_BY_JOINT[joint]
            fb = driver.get_latest_type2_feedback(motor_id, max_age=self.max_type2_age_sec)
            if fb is None:
                return None
            fbs[joint] = fb

        names = []
        positions = []
        velocities = []
        efforts = []

        for joint in JOINT_NAMES_FOR_PUBLISH:
            fb = fbs[joint]
            sign = float(self.signs[joint])
            zero = float(self.zeros[joint])

            q = sign * (float(fb.position) - zero)
            qdot = sign * float(fb.velocity)
            effort = sign * float(fb.torque)

            names.append(joint)
            positions.append(q)
            velocities.append(qdot)
            efforts.append(effort)

        for passive_joint, passive_q in PASSIVE_JOINT_STATES_FOR_PUBLISH.items():
            names.append(passive_joint)
            positions.append(float(passive_q))
            velocities.append(0.0)
            efforts.append(0.0)

        return names, positions, velocities, efforts

    def publish(self, driver, cmd):
        if not self.enabled:
            return

        state = None
        source_used = None

        if self.source in ("auto", "type2"):
            state = self._type2_state_from_driver(driver)
            if state is not None:
                source_used = "type2"
            elif self.source == "type2":
                self.type2_missing_count += 1
                return

        if state is None:
            state = self._commanded_state_from_cmd(cmd)
            source_used = "commanded"

        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(state[0])
        msg.position = [float(x) for x in state[1]]
        msg.velocity = [float(x) for x in state[2]]
        msg.effort = [float(x) for x in state[3]]

        self.pub.publish(msg)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

        self.publish_count += 1
        self.last_source = source_used

        if source_used == "type2":
            self.type2_publish_count += 1
        else:
            self.commanded_publish_count += 1

    def stats(self):
        return {
            "enabled": self.enabled,
            "topic": self.topic,
            "source": self.source,
            "max_type2_age_sec": self.max_type2_age_sec,
            "publish_count": self.publish_count,
            "commanded_publish_count": self.commanded_publish_count,
            "type2_publish_count": self.type2_publish_count,
            "type2_missing_count": self.type2_missing_count,
            "last_source": self.last_source,
        }

    def close(self):
        if not self.enabled:
            return

        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass


def setup_joint_state_publisher(args):
    global JOINT_STATE_PUBLISHER

    JOINT_STATE_PUBLISHER = ExecutorJointStatePublisher(
        enabled=bool(args.publish_joint_states),
        topic=args.joint_state_topic,
        source=args.joint_state_source,
        offset_json=args.offset_json,
        max_type2_age_sec=float(args.max_type2_age_sec),
    )


def maybe_publish_joint_states(driver, cmd):
    if JOINT_STATE_PUBLISHER is not None:
        JOINT_STATE_PUBLISHER.publish(driver, cmd)


def close_joint_state_publisher():
    global JOINT_STATE_PUBLISHER

    if JOINT_STATE_PUBLISHER is not None:
        JOINT_STATE_PUBLISHER.close()


def get_joint_state_publisher_stats():
    if JOINT_STATE_PUBLISHER is None:
        return {"enabled": False}
    return JOINT_STATE_PUBLISHER.stats()

def send_type1_point(driver, cmd, delay):
    for motor_id in ARM_MOTOR_IDS:
        c = cmd["motors"][str(motor_id)]
        driver.send_motion_control(
            motor_id=motor_id,
            position=float(c["position"]),
            velocity=float(c["velocity"]),
            kp=float(c["kp"]),
            kd=float(c["kd"]),
            torque=float(c["torque_ff"]),
        )
        if delay > 0:
            time.sleep(delay)

    maybe_publish_joint_states(driver, cmd)



def make_hold_command(last_cmd):
    hold = {
        "point_index": int(last_cmd.get("point_index", 0)),
        "time_from_start_sec": float(last_cmd.get("time_from_start_sec", 0.0)),
        "motors": {},
    }

    for motor_id_str, c in last_cmd["motors"].items():
        d = dict(c)
        d["velocity"] = 0.0
        hold["motors"][motor_id_str] = d

    return hold


def interpolate_command(c0, c1, t, sample_index):
    t0 = float(c0["time_from_start_sec"])
    t1 = float(c1["time_from_start_sec"])

    if t1 <= t0:
        alpha = 0.0
    else:
        alpha = max(0.0, min(1.0, (float(t) - t0) / (t1 - t0)))

    out = {
        "point_index": int(sample_index),
        "time_from_start_sec": float(t),
        "motors": {},
    }

    for motor_id_str in c0["motors"].keys():
        a = c0["motors"][motor_id_str]
        b = c1["motors"][motor_id_str]

        out["motors"][motor_id_str] = {
            "joint": a["joint"],
            "motor_id": int(a["motor_id"]),
            "position": float(a["position"]) + alpha * (float(b["position"]) - float(a["position"])),
            "velocity": float(a["velocity"]) + alpha * (float(b["velocity"]) - float(a["velocity"])),
            "kp": float(a["kp"]),
            "kd": float(a["kd"]),
            "torque_ff": float(a["torque_ff"]) + alpha * (float(b["torque_ff"]) - float(a["torque_ff"])),
        }

    return out


def resample_commands(commands, rate_hz):
    if not commands:
        return []

    if len(commands) == 1:
        return [commands[0]]

    rate_hz = float(rate_hz)
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    duration = float(commands[-1]["time_from_start_sec"])
    dt = 1.0 / rate_hz

    sample_times = []
    t = 0.0
    while t < duration - 1e-9:
        sample_times.append(t)
        t += dt

    if not sample_times or abs(sample_times[-1] - duration) > 1e-9:
        sample_times.append(duration)

    out = []
    seg = 0

    for sample_index, t in enumerate(sample_times):
        while seg < len(commands) - 2 and float(commands[seg + 1]["time_from_start_sec"]) < t - 1e-9:
            seg += 1

        c0 = commands[seg]
        c1 = commands[min(seg + 1, len(commands) - 1)]
        out.append(interpolate_command(c0, c1, t, sample_index))

    return out



def make_hold_command(last_cmd):
    hold = {
        "point_index": int(last_cmd.get("point_index", 0)),
        "time_from_start_sec": float(last_cmd.get("time_from_start_sec", 0.0)),
        "motors": {},
    }

    for motor_id_str, c in last_cmd["motors"].items():
        d = dict(c)
        d["velocity"] = 0.0
        hold["motors"][motor_id_str] = d

    return hold


def interpolate_command(c0, c1, t, sample_index):
    t0 = float(c0["time_from_start_sec"])
    t1 = float(c1["time_from_start_sec"])

    if t1 <= t0:
        alpha = 0.0
    else:
        alpha = max(0.0, min(1.0, (float(t) - t0) / (t1 - t0)))

    out = {
        "point_index": int(sample_index),
        "time_from_start_sec": float(t),
        "motors": {},
    }

    for motor_id_str in c0["motors"].keys():
        a = c0["motors"][motor_id_str]
        b = c1["motors"][motor_id_str]

        out["motors"][motor_id_str] = {
            "joint": a["joint"],
            "motor_id": int(a["motor_id"]),
            "position": float(a["position"]) + alpha * (float(b["position"]) - float(a["position"])),
            "velocity": float(a["velocity"]) + alpha * (float(b["velocity"]) - float(a["velocity"])),
            "kp": float(a["kp"]),
            "kd": float(a["kd"]),
            "torque_ff": float(a["torque_ff"]) + alpha * (float(b["torque_ff"]) - float(a["torque_ff"])),
        }

    return out


def resample_commands(commands, rate_hz):
    if not commands:
        return []

    if len(commands) == 1:
        return [commands[0]]

    rate_hz = float(rate_hz)
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    duration = float(commands[-1]["time_from_start_sec"])
    dt = 1.0 / rate_hz

    sample_times = []
    t = 0.0
    while t < duration - 1e-9:
        sample_times.append(t)
        t += dt

    if not sample_times or abs(sample_times[-1] - duration) > 1e-9:
        sample_times.append(duration)

    out = []
    seg = 0

    for sample_index, t in enumerate(sample_times):
        while seg < len(commands) - 2 and float(commands[seg + 1]["time_from_start_sec"]) < t - 1e-9:
            seg += 1

        c0 = commands[seg]
        c1 = commands[min(seg + 1, len(commands) - 1)]
        out.append(interpolate_command(c0, c1, t, sample_index))

    return out


def execute_real(args, commands):
    exec_commands = resample_commands(commands, float(args.rate))

    runtime = {
        "attempted": True,
        "ok": False,
        "can": args.can,
        "post_action": args.post_action,
        "raw_waypoint_count": len(commands),
        "resampled_type1_command_count": len(exec_commands),
        "type17_start_verify_sent": False,
        "type4_disable_sent": False,
        "type18_set_motion_mode_sent": False,
        "type3_enable_sent": False,
        "type1_sent": False,
        "type1_point_count_sent": 0,
        "hold_type1_count_sent": 0,
        "hold_elapsed_sec": 0.0,
        "exception": None,
        "keyboard_interrupt": False,
        "keyboard_interrupt_during_hold": False,
        "driver_stats": None,
        "motor_start_verify": None,
    }

    driver = SukineeSocketCANDriver(args.can)
    opened = False
    did_send_enable_or_type1 = False

    setup_joint_state_publisher(args)

    try:
        print()
        print("ARMED REAL EXECUTION requested.")
        print(f"Raw YAML waypoints: {len(commands)}")
        print(f"Resampled Type1 commands at {args.rate:.3f} Hz: {len(exec_commands)}")
        print("Opening SocketCAN driver...")
        driver.open()
        opened = True

        print("Verifying actual motor start positions using Type17 POS_INDEX=0x7019...")
        ok_start, start_details = verify_motor_start(
            driver=driver,
            commands=exec_commands,
            max_error_rad=float(args.max_start_error_rad),
            read_timeout=float(args.motor_read_timeout),
        )
        runtime["type17_start_verify_sent"] = True
        runtime["motor_start_verify"] = start_details

        print(f"  motor start max_abs_error_rad={start_details['max_abs_error_rad']:.6f}")
        if not ok_start:
            raise RuntimeError(
                "Actual motor start position does not match first trajectory waypoint. "
                "No Type4/Type3/Type1 command was sent after this failure."
            )

        print("Sending Type4 disable to Joint1-Joint6 before enabling motion-control path...")
        send_disable_all(driver, delay=float(args.inter_motor_delay))
        runtime["type4_disable_sent"] = True
        time.sleep(float(args.stage_delay))

        if args.skip_set_motion_mode:
            print("Skipping Type18 RUN_MODE_INDEX write because --skip-set-motion-mode was provided.")
        else:
            print("Sending Type18 RUN_MODE_INDEX=0x7005 -> motion-control mode to Joint1-Joint6...")
            send_set_motion_mode_all(driver, delay=float(args.inter_motor_delay))
            runtime["type18_set_motion_mode_sent"] = True
            time.sleep(float(args.stage_delay))

        print("Sending Type3 enable to Joint1-Joint6...")
        send_enable_all(driver, delay=float(args.inter_motor_delay))
        runtime["type3_enable_sent"] = True
        did_send_enable_or_type1 = True
        time.sleep(float(args.stage_delay))

        first = exec_commands[0]
        if args.prehold_sec > 0:
            print(f"Pre-holding first Type1 point for {args.prehold_sec:.3f} sec...")
            pre_end = time.monotonic() + float(args.prehold_sec)
            while time.monotonic() < pre_end:
                send_type1_point(driver, first, delay=float(args.inter_motor_delay))
                runtime["type1_sent"] = True
                time.sleep(max(0.0, 1.0 / float(args.rate)))

        print("Executing resampled Type1 trajectory...")
        t0 = time.monotonic()

        for idx, cmd in enumerate(exec_commands):
            target_t = t0 + float(cmd["time_from_start_sec"])
            while True:
                now = time.monotonic()
                if now >= target_t:
                    break
                time.sleep(min(0.002, target_t - now))

            send_type1_point(driver, cmd, delay=float(args.inter_motor_delay))
            runtime["type1_sent"] = True
            runtime["type1_point_count_sent"] += 1

            if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(exec_commands) - 1):
                print(f"  sent Type1 sample {idx + 1}/{len(exec_commands)} at t={cmd['time_from_start_sec']:.3f}s")

        if args.post_action == "hold":
            hold_cmd = make_hold_command(exec_commands[-1])
            hold_rate = float(args.hold_rate)
            if hold_rate <= 0:
                raise RuntimeError("--hold-rate must be positive when --post-action hold is used.")

            print()
            print("Post-action: HOLD")
            print("Holding final Type1 point:")
            print("  position = final motor position")
            print("  velocity = 0")
            print(f"  kp       = {args.preview_kp:.6f}")
            print(f"  kd       = {args.preview_kd:.6f}")
            print("  torque   = final gravity_ff")
            if args.max_hold_sec > 0:
                print(f"  max_hold_sec = {args.max_hold_sec:.3f}")
            else:
                print("  max_hold_sec = infinite; press Ctrl-C to leave hold and Type4 disable")

            hold_start = time.monotonic()
            hold_count = 0

            try:
                while True:
                    elapsed = time.monotonic() - hold_start
                    if args.max_hold_sec > 0 and elapsed >= float(args.max_hold_sec):
                        print(f"Max hold time reached: {elapsed:.3f} sec")
                        break

                    loop_t0 = time.monotonic()
                    send_type1_point(driver, hold_cmd, delay=float(args.inter_motor_delay))
                    runtime["type1_sent"] = True
                    hold_count += 1
                    runtime["hold_type1_count_sent"] = hold_count
                    runtime["hold_elapsed_sec"] = time.monotonic() - hold_start

                    if args.hold_print_every > 0 and hold_count % int(args.hold_print_every) == 0:
                        print(f"  holding final point: count={hold_count}, elapsed={runtime['hold_elapsed_sec']:.3f}s")

                    sleep_dt = max(0.0, (1.0 / hold_rate) - (time.monotonic() - loop_t0))
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)

            except KeyboardInterrupt:
                runtime["keyboard_interrupt_during_hold"] = True
                runtime["hold_elapsed_sec"] = time.monotonic() - hold_start
                print()
                print("Ctrl-C during HOLD. Leaving hold loop and sending Type4 disable in finally path.")

        else:
            print()
            print("Post-action: DISABLE")
            if args.final_hold_sec > 0:
                print(f"Final-holding last Type1 point for {args.final_hold_sec:.3f} sec before disable...")
                hold_cmd = make_hold_command(exec_commands[-1])
                end_t = time.monotonic() + float(args.final_hold_sec)
                while time.monotonic() < end_t:
                    send_type1_point(driver, hold_cmd, delay=float(args.inter_motor_delay))
                    runtime["type1_sent"] = True
                    time.sleep(max(0.0, 1.0 / float(args.rate)))

        runtime["ok"] = True
        print("REAL EXECUTION LOOP RESULT: PASS")

    except KeyboardInterrupt:
        runtime["keyboard_interrupt"] = True
        runtime["exception"] = "KeyboardInterrupt"
        print()
        print("KeyboardInterrupt received during trajectory execution. Sending Type4 disable in finally path.")

    except Exception as exc:
        runtime["exception"] = repr(exc)
        print()
        print(f"REAL EXECUTION LOOP RESULT: FAIL: {exc}")

    finally:
        if opened:
            if args.disable_on_exit and did_send_enable_or_type1:
                try:
                    print("Sending Type4 disable to Joint1-Joint6 on exit...")
                    send_disable_all(driver, delay=float(args.inter_motor_delay))
                    runtime["type4_disable_sent"] = True
                except Exception as exc:
                    runtime["exception_on_disable"] = repr(exc)
                    print(f"WARNING: failed to send Type4 disable on exit: {exc}")

            try:
                runtime["driver_stats"] = driver.get_stats()
            except Exception:
                pass

            try:
                driver.close()
            except Exception:
                pass

            try:
                close_joint_state_publisher()
            except Exception:
                pass

    return runtime

def write_json(path_text, payload):
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sukinee Type1 trajectory executor v1. Default is dry-run."
    )

    parser.add_argument("--trajectory-yaml", required=True)
    parser.add_argument("--limits-yaml", default=str(safety.DEFAULT_LIMITS_YAML))
    parser.add_argument("--offset-json", default=DEFAULT_OFFSET_JSON)
    parser.add_argument("--config-json", default=DEFAULT_CONFIG_JSON)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--current-q", default="")
    parser.add_argument("--max-start-error-rad", type=float, default=0.05)
    parser.add_argument("--preview-kp", type=float, default=0.0, help="Kp preview in dry-run; actual Kp in armed mode.")
    parser.add_argument("--preview-kd", type=float, default=0.0, help="Kd preview in dry-run; actual Kd in armed mode.")
    parser.add_argument("--skip-gravity-precheck", action="store_true")
    parser.add_argument("--output-json", default=DEFAULT_OUT)

    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm", default="")

    parser.add_argument("--enforce-real-guards", action="store_true")
    parser.add_argument("--max-real-duration-sec", type=float, default=8.0)
    parser.add_argument("--max-real-joint-motion-rad", type=float, default=0.80)
    parser.add_argument("--min-real-kp", type=float, default=0.1)
    parser.add_argument("--max-real-kp", type=float, default=20.0)
    parser.add_argument("--max-real-kd", type=float, default=1.0)

    parser.add_argument("--motor-read-timeout", type=float, default=0.6)
    parser.add_argument("--skip-set-motion-mode", action="store_true")
    parser.add_argument("--inter-motor-delay", type=float, default=0.002)
    parser.add_argument("--stage-delay", type=float, default=0.05)
    parser.add_argument("--prehold-sec", type=float, default=0.20)
    parser.add_argument("--final-hold-sec", type=float, default=0.20)
    parser.add_argument("--post-action", choices=["disable", "hold"], default="disable")
    parser.add_argument("--hold-rate", type=float, default=50.0)
    parser.add_argument("--hold-print-every", type=int, default=50)
    parser.add_argument("--max-hold-sec", type=float, default=0.0, help="0 means hold forever until Ctrl-C.")
    parser.add_argument("--disable-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--publish-joint-states", action="store_true", help="Publish /joint_states during armed real execution.")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--joint-state-source", choices=["auto", "commanded", "type2"], default="auto")
    parser.add_argument("--max-type2-age-sec", type=float, default=0.20)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rate <= 0:
        print("ERROR: --rate must be positive.")
        return 2

    if args.post_action == "hold" and args.hold_rate <= 0:
        print("ERROR: --hold-rate must be positive when --post-action hold is used.")
        return 2

    if args.max_hold_sec < 0:
        print("ERROR: --max-hold-sec must be >= 0.")
        return 2

    if args.max_type2_age_sec <= 0:
        print("ERROR: --max-type2-age-sec must be positive.")
        return 2

    if args.joint_state_source == "type2" and not args.publish_joint_states:
        print("ERROR: --joint-state-source type2 requires --publish-joint-states.")
        return 2

    if args.armed and args.confirm != CONFIRM_TEXT:
        print("ERROR: armed mode requires exact confirmation:")
        print(f"  --confirm {CONFIRM_TEXT}")
        return 2

    if args.armed:
        print("IMPORTANT REAL EXECUTION PRECONDITIONS:")
        print("  1. Stop zero_drag_rviz.launch.py before running armed executor.")
        print("  2. Stop sukinee_real_state_moveit_visualization.launch.py before armed execution to avoid CAN competition.")
        print("  3. Do not click MoveIt Execute in RViz.")
        print("  4. Keep the stop-all command ready in another terminal.")
        print("  5. This script will send Type1 commands when all checks pass.")
        print(f"  6. post_action={args.post_action}; hold means final position is actively maintained until Ctrl-C or max-hold-sec.")
        print(f"  7. publish_joint_states={args.publish_joint_states}; if enabled, executor publishes RViz /joint_states itself.")
        print()

    ok, report, context = load_and_check_trajectory(args)

    commands = []
    command_ranges = {}
    real_guard = {}
    runtime = {
        "attempted": False,
        "ok": False,
    }

    if ok:
        commands, command_ranges = compute_type1_commands(args, context)
        real_guard = apply_real_execution_guards(args, context, command_ranges, report)
        ok = report.ok

    if ok and args.armed:
        runtime = execute_real(args, commands)
        ok = bool(runtime.get("ok", False))

    summary = {
        "ok": bool(ok),
        "version": VERSION,
        "mode": "armed_real_type1" if args.armed else "dry_run",
        "armed": bool(args.armed),
        "trajectory_yaml": str(Path(args.trajectory_yaml).expanduser()),
        "point_count": len(context.get("times", [])),
        "duration_sec": (
            context.get("times", [0])[-1] - context.get("times", [0])[0]
            if context.get("times")
            else 0.0
        ),
        "rate": float(args.rate),
        "post_action": args.post_action,
        "hold_rate": float(args.hold_rate),
        "max_hold_sec": float(args.max_hold_sec),
        "raw_waypoint_count": len(commands),
        "resampled_command_count_at_rate": len(resample_commands(commands, args.rate)) if commands else 0,
        "kp": float(args.preview_kp),
        "kd": float(args.preview_kd),
        "safety_boundary": {
            "dry_run_only": not bool(args.armed),
            "socketcan_opened": bool(args.armed),
            "type17_start_verify_sent": bool(runtime.get("type17_start_verify_sent", False)),
            "type4_disable_sent": bool(runtime.get("type4_disable_sent", False)),
            "type18_set_motion_mode_sent": bool(runtime.get("type18_set_motion_mode_sent", False)),
            "type3_enable_sent": bool(runtime.get("type3_enable_sent", False)),
            "type1_sent": bool(runtime.get("type1_sent", False)),
            "type6_set_zero_sent": False,
            "save_motor_parameters": False,
            "change_can_id": False,
            "switch_protocol": False,
            "moveit_real_execution": False,
        },
        "observed_trajectory": context.get("observed", {}),
        "type1_command_ranges": command_ranges,
        "real_guard": real_guard,
        "runtime": runtime,
        "joint_state_publisher": get_joint_state_publisher_stats(),
        "report": report.as_dict(),
    }

    write_json(args.output_json, summary)
    print()
    print(f"Wrote executor v1 JSON: {Path(args.output_json).expanduser()}")

    if args.armed:
        print("ARMED EXECUTION RESULT:", "PASS" if ok else "FAIL")
    else:
        print("DRY-RUN RESULT:", "PASS" if ok else "FAIL")
        print("No CAN command was sent in dry-run mode.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
