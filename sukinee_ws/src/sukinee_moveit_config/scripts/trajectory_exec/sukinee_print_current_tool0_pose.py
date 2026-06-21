#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print current tool0 pose relative to base_link and generate command templates.

Safety boundary:
- Reads TF only.
- Does NOT call MoveIt.
- Does NOT plan trajectory.
- Does NOT execute trajectory.
- Does NOT open SocketCAN.
- Does NOT send motor commands.
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def quaternion_to_rpy(x: float, y: float, z: float, w: float):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class CurrentTool0PoseNode(Node):
    def __init__(self, args):
        super().__init__("sukinee_print_current_tool0_pose")
        self.args = args
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

    def run(self) -> int:
        self.get_logger().info(
            f"Waiting for TF: {self.args.frame_id} -> {self.args.eef_link}"
        )

        deadline = self.get_clock().now().nanoseconds * 1e-9 + self.args.timeout

        while rclpy.ok():
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec > deadline:
                self.get_logger().error("Timeout waiting for TF.")
                return 2

            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                tf = self.buffer.lookup_transform(
                    self.args.frame_id,
                    self.args.eef_link,
                    rclpy.time.Time(),
                )
                break
            except Exception:
                continue

        t = tf.transform.translation
        q = tf.transform.rotation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)

        target_x = t.x + self.args.dx
        target_y = t.y + self.args.dy
        target_z = t.z + self.args.dz

        print("")
        print("Current tool0 pose:")
        print(f"  frame_id: {self.args.frame_id}")
        print(f"  eef_link: {self.args.eef_link}")
        print(f"  x: {t.x:.6f}")
        print(f"  y: {t.y:.6f}")
        print(f"  z: {t.z:.6f}")
        print(f"  roll:  {roll:.6f}")
        print(f"  pitch: {pitch:.6f}")
        print(f"  yaw:   {yaw:.6f}")

        print("")
        print("Offset target pose:")
        print(f"  dx: {self.args.dx:.6f}")
        print(f"  dy: {self.args.dy:.6f}")
        print(f"  dz: {self.args.dz:.6f}")
        print(f"  target_x: {target_x:.6f}")
        print(f"  target_y: {target_y:.6f}")
        print(f"  target_z: {target_z:.6f}")
        print(f"  target_roll:  {roll:.6f}")
        print(f"  target_pitch: {pitch:.6f}")
        print(f"  target_yaw:   {yaw:.6f}")

        print("")
        print("Preview command:")
        print(f"""python3 /home/zzj/sukinee_ws/src/sukinee_moveit_config/scripts/trajectory_exec/sukinee_preview_target_tool0_tf.py \\
  --frame-id {self.args.frame_id} \\
  --child-frame-id target_tool0_preview \\
  --target-x {target_x:.6f} \\
  --target-y {target_y:.6f} \\
  --target-z {target_z:.6f} \\
  --target-roll {roll:.6f} \\
  --target-pitch {pitch:.6f} \\
  --target-yaw {yaw:.6f}""")

        print("")
        print("Plan export command:")
        print(f"""python3 /home/zzj/sukinee_ws/src/sukinee_moveit_config/scripts/trajectory_exec/sukinee_plan_to_pose_export.py \\
  --frame-id {self.args.frame_id} \\
  --target-x {target_x:.6f} \\
  --target-y {target_y:.6f} \\
  --target-z {target_z:.6f} \\
  --target-roll {roll:.6f} \\
  --target-pitch {pitch:.6f} \\
  --target-yaw {yaw:.6f} \\
  --position-tolerance {self.args.position_tolerance:.6f} \\
  --orientation-tolerance {self.args.orientation_tolerance:.6f} \\
  --output-yaml {self.args.output_yaml}""")

        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print current tool0 pose and generate preview/planning commands."
    )

    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--eef-link", default="tool0")
    parser.add_argument("--timeout", type=float, default=5.0)

    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.02)

    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.20)

    parser.add_argument(
        "--output-yaml",
        default="/home/zzj/sukinee_ws/trajectories/plan_auto_from_current.yaml",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rclpy.init()
    node = CurrentTool0PoseNode(args)

    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
