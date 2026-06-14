#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


ARM_JOINT_NAMES = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]

LOCK_JOINT_NAMES = [
    "Joint7",
    "left_finger",
    "right_finger",
]


def build_arm_model(urdf_path: Path):
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    q_full_neutral = pin.neutral(full_model)

    lock_joint_ids = []
    for name in LOCK_JOINT_NAMES:
        joint_id = full_model.getJointId(name)
        if joint_id >= len(full_model.joints):
            raise RuntimeError(f"Joint not found in URDF: {name}")
        lock_joint_ids.append(joint_id)

    arm_model = pin.buildReducedModel(full_model, lock_joint_ids, q_full_neutral)
    return arm_model


def make_q_from_arm_positions(model, positions_by_name):
    q = pin.neutral(model)

    for joint_name in ARM_JOINT_NAMES:
        angle = positions_by_name[joint_name]

        joint_id = model.getJointId(joint_name)
        if joint_id >= len(model.joints):
            raise RuntimeError(f"Joint not found in reduced model: {joint_name}")

        joint = model.joints[joint_id]

        if joint.nq == 1:
            q[joint.idx_q] = angle

        elif joint.nq == 2 and joint.nv == 1:
            q[joint.idx_q] = np.cos(angle)
            q[joint.idx_q + 1] = np.sin(angle)

        else:
            raise RuntimeError(
                f"Unsupported joint representation for {joint_name}: "
                f"nq={joint.nq}, nv={joint.nv}"
            )

    return q


def make_v_from_arm_velocities(model, velocities_by_name):
    v = np.zeros(model.nv)

    for joint_name in ARM_JOINT_NAMES:
        velocity = velocities_by_name.get(joint_name, 0.0)

        joint_id = model.getJointId(joint_name)
        if joint_id >= len(model.joints):
            raise RuntimeError(f"Joint not found in reduced model: {joint_name}")

        joint = model.joints[joint_id]

        if joint.nv != 1:
            raise RuntimeError(
                f"Unsupported joint velocity representation for {joint_name}: "
                f"nv={joint.nv}"
            )

        v[joint.idx_v] = velocity

    return v


def vector_to_arm_order(model, vector):
    values = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = model.getJointId(joint_name)
        joint = model.joints[joint_id]
        values.append(float(vector[joint.idx_v]))

    return values


class GazeboGravityCompNode(Node):
    def __init__(self, args):
        super().__init__("gz_gravity_comp_node")

        self.urdf_path = Path(args.urdf)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.model = build_arm_model(self.urdf_path)
        self.data = self.model.createData()

        self.state_topic = args.state_topic
        self.command_topic = args.command_topic

        self.dry_run = args.dry_run

        if len(args.gravity_scales) != 6:
            raise ValueError("--gravity-scales must have exactly 6 values")

        if len(args.damping) != 6:
            raise ValueError("--damping must have exactly 6 values")

        if len(args.joint_signs) != 6:
            raise ValueError("--joint-signs must have exactly 6 values")

        if len(args.effort_limits) != 6:
            raise ValueError("--effort-limits must have exactly 6 values")

        if len(args.torque_rate_limits) != 6:
            raise ValueError("--torque-rate-limits must have exactly 6 values")

        self.gravity_scales = np.array(args.gravity_scales, dtype=float)
        self.damping = np.array(args.damping, dtype=float)
        self.joint_signs = np.array(args.joint_signs, dtype=float)
        self.effort_limits = np.array(args.effort_limits, dtype=float)
        self.torque_rate_limits = np.array(args.torque_rate_limits, dtype=float)
        self.startup_ramp_time = float(args.startup_ramp_time)
        self.state_timeout = float(args.state_timeout)
        self.max_velocity = float(args.max_velocity)
        self.dt = 1.0 / float(args.rate)
        self.log_interval = float(args.log_interval)

        self.start_time = self.get_clock().now()
        self.command_start_time = None
        self.last_tau_cmd = np.zeros(6, dtype=float)
        self.mode = "ARMED"
        self.fault_reason = ""

        self.latest_positions = None
        self.latest_velocities = None
        self.latest_state_time = None
        self.last_status_log_time = self.get_clock().now()
        self.last_warning_log_time = None

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.gravity_debug_pub = self.create_publisher(
            Float64MultiArray,
            "/sukinee_gravity/tau_g",
            10,
        )

        self.command_debug_pub = self.create_publisher(
            Float64MultiArray,
            "/sukinee_gravity/tau_cmd",
            10,
        )

        self.mode_pub = self.create_publisher(
            String,
            "/sukinee_gravity/mode",
            10,
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.state_topic,
            self.joint_state_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / args.rate, self.control_loop)

        self.get_logger().info("Gazebo gravity compensation node started")
        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(f"Pinocchio reduced model nq={self.model.nq}, nv={self.model.nv}")
        self.get_logger().info(f"state_topic: {self.state_topic}")
        self.get_logger().info(f"command_topic: {self.command_topic}")
        self.get_logger().info(f"rate: {args.rate} Hz")
        self.get_logger().info(f"log_interval: {self.log_interval} s")
        self.get_logger().info(f"gravity_scales: {self.gravity_scales.tolist()}")
        self.get_logger().info(f"damping: {self.damping.tolist()}")
        self.get_logger().info(f"joint_signs: {self.joint_signs.tolist()}")
        self.get_logger().info(f"effort_limits: {self.effort_limits.tolist()}")
        self.get_logger().info(f"torque_rate_limits: {self.torque_rate_limits.tolist()}")
        self.get_logger().info(f"startup_ramp_time: {self.startup_ramp_time}")
        self.get_logger().info(f"state_timeout: {self.state_timeout}")
        self.get_logger().info(f"max_velocity: {self.max_velocity}")
        self.get_logger().info(f"initial_mode: {self.mode}")

        if self.dry_run:
            self.get_logger().warn("DRY RUN enabled: commands will NOT be published to the controller")

    def joint_state_callback(self, msg: JointState):
        missing = [name for name in ARM_JOINT_NAMES if name not in msg.name]
        if missing:
            self.get_logger().warn(f"Missing joints in /joint_states: {missing}")
            return

        positions = {}
        velocities = {}

        for joint_name in ARM_JOINT_NAMES:
            idx = msg.name.index(joint_name)

            if idx >= len(msg.position):
                self.get_logger().warn(f"No position for {joint_name}")
                return

            positions[joint_name] = float(msg.position[idx])

            if idx < len(msg.velocity):
                velocities[joint_name] = float(msg.velocity[idx])
            else:
                velocities[joint_name] = 0.0

        self.latest_positions = positions
        self.latest_velocities = velocities
        self.latest_state_time = self.get_clock().now()

    def control_loop(self):
        self.publish_mode()

        if self.mode == "FAULT":
            self.publish_safe_zero(self.fault_reason)
            return

        if self.latest_positions is None:
            self.set_mode("ARMED", "Waiting for joint states")
            self.publish_safe_zero("Waiting for joint states")
            return

        now = self.get_clock().now()
        if self.latest_state_time is None:
            self.enter_fault("No joint state timestamp yet")
            return

        state_age = (now - self.latest_state_time).nanoseconds * 1e-9
        if state_age > self.state_timeout:
            self.enter_fault(f"Joint state timeout: {state_age:.3f}s")
            return

        if not self.dry_run and self.command_pub.get_subscription_count() == 0:
            self.set_mode("ARMED", "Waiting for effort controller subscriber")
            self.publish_safe_zero("No subscriber on effort command topic")
            return

        if self.mode == "ARMED":
            self.set_mode("DRAG", "Effort controller subscriber detected")
            self.command_start_time = now
            self.last_tau_cmd = np.zeros(6, dtype=float)

        q = make_q_from_arm_positions(self.model, self.latest_positions)
        v = make_v_from_arm_velocities(self.model, self.latest_velocities)

        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)

        tau_g_arm_order = np.array(vector_to_arm_order(self.model, tau_g), dtype=float)
        v_arm_order = np.array(vector_to_arm_order(self.model, v), dtype=float)

        if not np.all(np.isfinite(tau_g_arm_order)) or not np.all(np.isfinite(v_arm_order)):
            self.enter_fault("Non-finite gravity or velocity value")
            return

        max_abs_velocity = float(np.max(np.abs(v_arm_order)))
        if max_abs_velocity > self.max_velocity:
            joint_index = int(np.argmax(np.abs(v_arm_order)))
            joint_name = ARM_JOINT_NAMES[joint_index]
            joint_velocity = float(v_arm_order[joint_index])
            self.enter_fault(
                f"Velocity limit exceeded on {joint_name}: "
                f"{joint_velocity:.3f} rad/s, max_abs={max_abs_velocity:.3f} rad/s"
            )
            return

        tau_cmd_arm_order = (
            self.joint_signs * self.gravity_scales * tau_g_arm_order
            - self.damping * v_arm_order
        )

        if self.startup_ramp_time > 0.0:
            now = self.get_clock().now()
            if self.dry_run:
                elapsed = (now - self.start_time).nanoseconds * 1e-9
            elif self.command_pub.get_subscription_count() > 0:
                if self.command_start_time is None:
                    self.command_start_time = now
                    self.last_tau_cmd = np.zeros(6, dtype=float)
                elapsed = (now - self.command_start_time).nanoseconds * 1e-9
            else:
                elapsed = 0.0

            ramp = min(1.0, max(0.0, elapsed / self.startup_ramp_time))
            tau_cmd_arm_order *= ramp

        max_delta = self.torque_rate_limits * self.dt
        tau_cmd_arm_order = np.clip(
            tau_cmd_arm_order,
            self.last_tau_cmd - max_delta,
            self.last_tau_cmd + max_delta,
        )

        tau_cmd_arm_order = np.clip(
            tau_cmd_arm_order,
            -self.effort_limits,
            self.effort_limits,
        )

        self.last_tau_cmd = tau_cmd_arm_order.copy()

        gravity_debug_msg = Float64MultiArray()
        gravity_debug_msg.data = tau_g_arm_order.tolist()
        self.gravity_debug_pub.publish(gravity_debug_msg)

        command_debug_msg = Float64MultiArray()
        command_debug_msg.data = tau_cmd_arm_order.tolist()
        self.command_debug_pub.publish(command_debug_msg)

        if not self.dry_run:
            command_msg = Float64MultiArray()
            command_msg.data = tau_cmd_arm_order.tolist()
            self.command_pub.publish(command_msg)

        if self.should_log_status():
            q_list = [self.latest_positions[name] for name in ARM_JOINT_NAMES]
            dq_list = [self.latest_velocities.get(name, 0.0) for name in ARM_JOINT_NAMES]

            self.get_logger().info(
                "q="
                + np.array2string(np.array(q_list), precision=3, suppress_small=True)
                + " dq="
                + np.array2string(np.array(dq_list), precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "tau_g="
                + np.array2string(tau_g_arm_order, precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "tau_cmd="
                + np.array2string(tau_cmd_arm_order, precision=3, suppress_small=True)
            )

    def should_log_status(self):
        if self.log_interval <= 0.0:
            return False

        now = self.get_clock().now()
        elapsed = (now - self.last_status_log_time).nanoseconds * 1e-9
        if elapsed < self.log_interval:
            return False

        self.last_status_log_time = now
        return True

    def publish_zero_once(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.command_pub.publish(msg)

    def set_mode(self, mode, reason):
        if self.mode == mode:
            return

        old_mode = self.mode
        self.mode = mode
        self.get_logger().warn(f"Mode transition: {old_mode} -> {mode}: {reason}")

    def enter_fault(self, reason):
        self.fault_reason = reason
        self.set_mode("FAULT", reason)
        self.publish_safe_zero(reason)

    def publish_mode(self):
        msg = String()
        if self.mode == "FAULT" and self.fault_reason:
            msg.data = f"{self.mode}: {self.fault_reason}"
        else:
            msg.data = self.mode
        self.mode_pub.publish(msg)

    def publish_safe_zero(self, reason):
        self.last_tau_cmd = np.zeros(6, dtype=float)

        if not self.dry_run:
            self.publish_zero_once()

        if self.should_log_warning():
            self.get_logger().warn(f"{reason}; publishing zero effort")

    def should_log_warning(self):
        if self.log_interval <= 0.0:
            return False

        now = self.get_clock().now()
        if self.last_warning_log_time is None:
            self.last_warning_log_time = now
            return True

        elapsed = (now - self.last_warning_log_time).nanoseconds * 1e-9
        if elapsed < self.log_interval:
            return False

        self.last_warning_log_time = now
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gazebo-only realtime gravity compensation for Sukinee Joint1-Joint6."
    )

    parser.add_argument(
        "--urdf",
        default="/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf",
        help="URDF file used by Pinocchio.",
    )

    parser.add_argument(
        "--state-topic",
        default="/joint_states",
        help="JointState topic.",
    )

    parser.add_argument(
        "--command-topic",
        default="/arm_effort_controller/commands",
        help="Effort command topic for arm_effort_controller.",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=150.0,
        help="Control loop rate in Hz.",
    )

    parser.add_argument(
        "--log-interval",
        type=float,
        default=0.1,
        help="Seconds between status log prints. Use 0 to disable periodic status logs.",
    )

    parser.add_argument(
        "--gravity-scales",
        nargs=6,
        type=float,
        default=[0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Per-joint gravity scale for Joint1-Joint6.",
    )

    parser.add_argument(
        "--damping",
        nargs=6,
        type=float,
        default=[0.05, 0.25, 0.24, 0.24, 1.20, 1.20],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Per-joint velocity damping for Joint1-Joint6.",
    )

    parser.add_argument(
        "--joint-signs",
        nargs=6,
        type=float,
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Per-joint torque sign for Joint1-Joint6.",
    )

    parser.add_argument(
        "--effort-limits",
        nargs=6,
        type=float,
        default=[0.2, 1.2, 2.0, 0.8, 0.22, 0.22],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Safety effort limits for Joint1-Joint6.",
    )

    parser.add_argument(
        "--torque-rate-limits",
        nargs=6,
        type=float,
        default=[1.0, 30.0, 30.0, 20.0, 1.5, 1.5],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Per-joint torque slew-rate limits in Nm/s.",
    )

    parser.add_argument(
        "--startup-ramp-time",
        type=float,
        default=0.0,
        help="Seconds used to ramp torque from zero after the controller subscribes.",
    )

    parser.add_argument(
        "--state-timeout",
        type=float,
        default=0.25,
        help="Publish zero effort if /joint_states is older than this many seconds.",
    )

    parser.add_argument(
        "--max-velocity",
        type=float,
        default=3.14,
        help="Publish zero effort if any arm joint velocity exceeds this rad/s. Keep this slightly above the URDF velocity limit.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print torques, but do not publish commands.",
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    rclpy.init()
    node = GazeboGravityCompNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.get_logger().warn("Stopping gravity compensation node, publishing zero effort once")
            try:
                node.publish_zero_once()
                rclpy.spin_once(node, timeout_sec=0.1)
            except Exception as exc:
                node.get_logger().warn(f"Failed to publish zero effort on shutdown: {exc}")
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()


if __name__ == "__main__":
    main()