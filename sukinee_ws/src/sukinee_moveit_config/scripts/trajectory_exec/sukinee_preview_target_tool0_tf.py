#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preview a target tool0 pose in RViz by publishing a TF frame.

This script is for visualization only.

Safety boundary:
- Publishes TF only.
- Does NOT call MoveIt.
- Does NOT plan trajectory.
- Does NOT execute trajectory.
- Does NOT open SocketCAN.
- Does NOT send Type1 / Type4 / motor commands.

Typical usage:
  python3 sukinee_preview_target_tool0_tf.py \
    --frame-id base_link \
    --child-frame-id target_tool0_preview \
    --target-x -0.192 \
    --target-y -0.032 \
    --target-z 0.185 \
    --target-roll -1.601 \
    --target-pitch -0.006 \
    --target-yaw 1.727

Then open RViz and enable TF display.
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


def rpy_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert roll/pitch/yaw in radians to quaternion xyzw."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return qx, qy, qz, qw


class TargetTool0PreviewNode(Node):
    def __init__(self, args):
        super().__init__("sukinee_preview_target_tool0_tf")

        self.args = args
        self.broadcaster = StaticTransformBroadcaster(self)

        self.transform = self._make_transform()
        self.broadcaster.sendTransform(self.transform)

        self.get_logger().info("Published target tool0 preview TF.")
        self.get_logger().info(
            f"{args.frame_id} -> {args.child_frame_id}: "
            f"xyz=({args.target_x:.6f}, {args.target_y:.6f}, {args.target_z:.6f}), "
            f"rpy=({args.target_roll:.6f}, {args.target_pitch:.6f}, {args.target_yaw:.6f})"
        )
        self.get_logger().info("Keep this node running while viewing RViz. Press Ctrl-C to stop.")

    def _make_transform(self) -> TransformStamped:
        qx, qy, qz, qw = rpy_to_quaternion(
            self.args.target_roll,
            self.args.target_pitch,
            self.args.target_yaw,
        )

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.args.frame_id
        t.child_frame_id = self.args.child_frame_id

        t.transform.translation.x = float(self.args.target_x)
        t.transform.translation.y = float(self.args.target_y)
        t.transform.translation.z = float(self.args.target_z)

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a target_tool0_preview TF frame for RViz visualization."
    )

    parser.add_argument(
        "--frame-id",
        default="base_link",
        help="Parent frame for target pose. Default: base_link.",
    )

    parser.add_argument(
        "--child-frame-id",
        default="target_tool0_preview",
        help="Child frame name to publish. Default: target_tool0_preview.",
    )

    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-z", type=float, required=True)
    parser.add_argument("--target-roll", type=float, required=True)
    parser.add_argument("--target-pitch", type=float, required=True)
    parser.add_argument("--target-yaw", type=float, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rclpy.init()
    node = TargetTool0PreviewNode(args)

    try:
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
