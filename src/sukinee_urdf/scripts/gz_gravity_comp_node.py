#!/usr/bin/env python3

from pathlib import Path
import argparse
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

        self.torque_gain = args.torque_gain
        self.kd = args.kd
        self.sign = args.sign
        self.dry_run = args.dry_run

        if len(args.effort_limits) != 6:
            raise ValueError("--effort-limits must have exactly 6 values")

        self.effort_limits = np.array(args.effort_limits, dtype=float)

        self.latest_positions = None
        self.latest_velocities = None
        self.print_count = 0

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
        self.get_logger().info(f"torque_gain: {self.torque_gain}")
        self.get_logger().info(f"kd: {self.kd}")
        self.get_logger().info(f"sign: {self.sign}")
        self.get_logger().info(f"effort_limits: {self.effort_limits.tolist()}")

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

    def control_loop(self):
        if self.latest_positions is None:
            return

        q = make_q_from_arm_positions(self.model, self.latest_positions)
        v = make_v_from_arm_velocities(self.model, self.latest_velocities)

        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)

        tau_cmd = self.sign * self.torque_gain * tau_g - self.kd * v

        tau_cmd_arm_order = np.array(vector_to_arm_order(self.model, tau_cmd), dtype=float)
        tau_g_arm_order = np.array(vector_to_arm_order(self.model, tau_g), dtype=float)

        tau_cmd_arm_order = np.clip(
            tau_cmd_arm_order,
            -self.effort_limits,
            self.effort_limits,
        )

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

        self.print_count += 1
        if self.print_count % 60 == 0:
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

    def publish_zero_once(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.command_pub.publish(msg)


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
        default=60.0,
        help="Control loop rate in Hz.",
    )

    parser.add_argument(
        "--torque-gain",
        type=float,
        default=0.3,
        help="Global gravity torque scale. Start small: 0.3, then 0.5, 0.8, 1.0.",
    )

    parser.add_argument(
        "--kd",
        type=float,
        default=0.05,
        help="Global velocity damping gain.",
    )

    parser.add_argument(
        "--sign",
        type=float,
        default=1.0,
        help="Global torque sign. Use -1.0 only if all gravity torques are reversed.",
    )

    parser.add_argument(
        "--effort-limits",
        nargs=6,
        type=float,
        default=[0.5, 0.8, 1.2, 0.6, 0.3, 0.3],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Safety effort limits for Joint1-Joint6.",
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
        node.get_logger().warn("Stopping gravity compensation node, publishing zero effort once")
        node.publish_zero_once()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()