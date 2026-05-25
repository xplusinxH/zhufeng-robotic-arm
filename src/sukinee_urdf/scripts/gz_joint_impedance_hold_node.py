#!/usr/bin/env python3

from pathlib import Path
import argparse
import math
import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


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


def wrap_to_pi(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


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
            q[joint.idx_q] = math.cos(angle)
            q[joint.idx_q + 1] = math.sin(angle)

        else:
            raise RuntimeError(
                f"Unsupported joint representation for {joint_name}: "
                f"nq={joint.nq}, nv={joint.nv}"
            )

    return q


def make_nv_vector_from_arm_values(model, values_by_name):
    v = np.zeros(model.nv)

    for joint_name in ARM_JOINT_NAMES:
        value = values_by_name.get(joint_name, 0.0)

        joint_id = model.getJointId(joint_name)
        if joint_id >= len(model.joints):
            raise RuntimeError(f"Joint not found in reduced model: {joint_name}")

        joint = model.joints[joint_id]

        if joint.nv != 1:
            raise RuntimeError(
                f"Unsupported joint velocity representation for {joint_name}: "
                f"nv={joint.nv}"
            )

        v[joint.idx_v] = value

    return v


def vector_to_arm_order(model, vector):
    values = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = model.getJointId(joint_name)
        joint = model.joints[joint_id]
        values.append(float(vector[joint.idx_v]))

    return np.array(values, dtype=float)


class GazeboJointImpedanceHoldNode(Node):
    def __init__(self, args):
        super().__init__("gz_joint_impedance_hold_node")

        self.urdf_path = Path(args.urdf)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.model = build_arm_model(self.urdf_path)
        self.data = self.model.createData()

        self.state_topic = args.state_topic
        self.command_topic = args.command_topic

        self.torque_gain = args.torque_gain
        self.kp = args.kp
        self.kd = args.kd
        self.sign = args.sign
        self.error_clip = abs(args.error_clip)
        self.dry_run = args.dry_run
        self.rate = args.rate
        self.torque_rate_limit = abs(args.torque_rate_limit)

        if len(args.effort_limits) != 6:
            raise ValueError("--effort-limits must have exactly 6 values")

        self.effort_limits = np.array(args.effort_limits, dtype=float)
        self.control_joint_names = list(args.control_joints)
        for name in self.control_joint_names:
            if name not in ARM_JOINT_NAMES:
                raise ValueError(
                    f"Unsupported control joint: {name}. "
                    f"Allowed joints: {ARM_JOINT_NAMES}"
                )

        self.control_joint_indices = [
            ARM_JOINT_NAMES.index(name)
            for name in self.control_joint_names
        ]

        if args.target is not None:
            if len(args.target) != 6:
                raise ValueError("--target must have exactly 6 values")
            self.q_target = {
                name: float(value)
                for name, value in zip(ARM_JOINT_NAMES, args.target)
            }
            self.target_captured = True
        else:
            self.q_target = None
            self.target_captured = False

        self.latest_positions = None
        self.latest_velocities = None
        self.last_tau_cmd = None
        self.print_count = 0

        self.command_pub = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.tau_g_pub = self.create_publisher(
            Float64MultiArray,
            "/sukinee_impedance/tau_g",
            10,
        )

        self.tau_cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/sukinee_impedance/tau_cmd",
            10,
        )

        self.error_pub = self.create_publisher(
            Float64MultiArray,
            "/sukinee_impedance/error",
            10,
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.state_topic,
            self.joint_state_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / self.rate, self.control_loop)

        self.get_logger().info("Gazebo joint impedance hold node started")
        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(f"Pinocchio reduced model nq={self.model.nq}, nv={self.model.nv}")
        self.get_logger().info(f"state_topic: {self.state_topic}")
        self.get_logger().info(f"command_topic: {self.command_topic}")
        self.get_logger().info(f"torque_gain: {self.torque_gain}")
        self.get_logger().info(f"kp: {self.kp}")
        self.get_logger().info(f"kd: {self.kd}")
        self.get_logger().info(f"sign: {self.sign}")
        self.get_logger().info(f"error_clip: ±{self.error_clip} rad")
        self.get_logger().info(f"effort_limits: {self.effort_limits.tolist()}")
        self.get_logger().info(f"control_joints: {self.control_joint_names}")
        self.get_logger().info(f"torque_rate_limit: {self.torque_rate_limit} Nm/s")

        if self.target_captured:
            self.get_logger().info(f"Using target from command line: {self.q_target}")
        else:
            self.get_logger().info("Target will be captured from the first valid /joint_states message")

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

        if self.q_target is None:
            self.q_target = dict(positions)
            self.target_captured = True
            self.get_logger().info("Captured q_target from current /joint_states:")
            for name in ARM_JOINT_NAMES:
                self.get_logger().info(f"  {name}: {self.q_target[name]:.6f} rad")

    def compute_error_by_name(self):
        error_by_name = {}

        for name in ARM_JOINT_NAMES:
            raw_error = self.q_target[name] - self.latest_positions[name]
            error = wrap_to_pi(raw_error)

            if error > self.error_clip:
                error = self.error_clip
            elif error < -self.error_clip:
                error = -self.error_clip

            error_by_name[name] = error

        return error_by_name

    def apply_rate_limit(self, tau_cmd):
        if self.last_tau_cmd is None:
            self.last_tau_cmd = tau_cmd.copy()
            return tau_cmd

        max_delta = self.torque_rate_limit / self.rate

        delta = tau_cmd - self.last_tau_cmd
        delta = np.clip(delta, -max_delta, max_delta)

        limited_tau = self.last_tau_cmd + delta
        self.last_tau_cmd = limited_tau.copy()

        return limited_tau

    def control_loop(self):
        if self.latest_positions is None or self.q_target is None:
            return

        q = make_q_from_arm_positions(self.model, self.latest_positions)
        dq = make_nv_vector_from_arm_values(self.model, self.latest_velocities)

        error_by_name = self.compute_error_by_name()
        error_vec = make_nv_vector_from_arm_values(self.model, error_by_name)

        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)
        tau_pd = self.kp * error_vec - self.kd * dq

        tau_cmd_vec = self.sign * self.torque_gain * tau_g + tau_pd

        tau_g_arm = vector_to_arm_order(self.model, tau_g)
        tau_pd_arm = vector_to_arm_order(self.model, tau_pd)
        tau_cmd_arm = vector_to_arm_order(self.model, tau_cmd_vec)
        error_arm = np.array([error_by_name[name] for name in ARM_JOINT_NAMES], dtype=float)

        tau_cmd_arm = np.clip(
            tau_cmd_arm,
            -self.effort_limits,
            self.effort_limits,
        )

        tau_cmd_arm = self.apply_rate_limit(tau_cmd_arm)

        tau_g_msg = Float64MultiArray()
        tau_g_msg.data = tau_g_arm.tolist()
        self.tau_g_pub.publish(tau_g_msg)

        tau_cmd_msg = Float64MultiArray()
        tau_cmd_msg.data = tau_cmd_arm.tolist()
        self.tau_cmd_pub.publish(tau_cmd_msg)

        error_msg = Float64MultiArray()
        error_msg.data = error_arm.tolist()
        self.error_pub.publish(error_msg)

        if not self.dry_run:
            command_msg = Float64MultiArray()
            command_msg.data = [
                float(tau_cmd_arm[i])
                for i in self.control_joint_indices
            ]
            self.command_pub.publish(command_msg)

        self.print_count += 1
        if self.print_count % 60 == 0:
            q_list = [self.latest_positions[name] for name in ARM_JOINT_NAMES]
            dq_list = [self.latest_velocities.get(name, 0.0) for name in ARM_JOINT_NAMES]
            target_list = [self.q_target[name] for name in ARM_JOINT_NAMES]

            self.get_logger().info(
                "q="
                + np.array2string(np.array(q_list), precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "target="
                + np.array2string(np.array(target_list), precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "dq="
                + np.array2string(np.array(dq_list), precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "error="
                + np.array2string(error_arm, precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "tau_g="
                + np.array2string(tau_g_arm, precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "tau_pd="
                + np.array2string(tau_pd_arm, precision=3, suppress_small=True)
            )
            self.get_logger().info(
                "tau_cmd="
                + np.array2string(tau_cmd_arm, precision=3, suppress_small=True)
            )

    def publish_zero_once(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * len(self.control_joint_names)
        self.command_pub.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gazebo-only joint impedance hold controller for Sukinee Joint1-Joint6."
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
        default=60.0,
        help="Control loop rate in Hz.",
    )

    parser.add_argument(
        "--torque-gain",
        type=float,
        default=1.0,
        help="Gravity torque scale.",
    )

    parser.add_argument(
        "--kp",
        type=float,
        default=1.5,
        help="Joint-space proportional stiffness gain.",
    )

    parser.add_argument(
        "--kd",
        type=float,
        default=0.4,
        help="Joint-space damping gain.",
    )

    parser.add_argument(
        "--sign",
        type=float,
        default=1.0,
        help="Global torque sign. Use -1.0 only if all gravity torques are reversed.",
    )

    parser.add_argument(
        "--error-clip",
        type=float,
        default=0.20,
        help="Maximum absolute joint position error used by PD term.",
    )

    parser.add_argument(
        "--effort-limits",
        nargs=6,
        type=float,
        default=[0.5, 1.5, 2.5, 1.0, 0.5, 0.5],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Safety effort limits for Joint1-Joint6.",
    )
    parser.add_argument(
        "--control-joints",
        nargs="+",
        default=ARM_JOINT_NAMES,
        help=(
            "Joints to publish commands for. "
            "Example: --control-joints Joint1 Joint2 Joint3 Joint4"
        ),
    )
    parser.add_argument(
        "--torque-rate-limit",
        type=float,
        default=10.0,
        help="Maximum torque change rate per joint in Nm/s.",
    )

    parser.add_argument(
        "--target",
        nargs=6,
        type=float,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Optional target joint position for Joint1-Joint6. If omitted, capture current pose.",
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
    node = GazeboJointImpedanceHoldNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().warn("Stopping impedance hold node, publishing zero effort once")
        node.publish_zero_once()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()