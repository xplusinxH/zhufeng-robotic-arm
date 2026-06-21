import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

import yaml

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory


DEFAULT_EXPECTED_JOINTS = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]


def duration_to_sec(duration_msg) -> float:
    return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9


def reorder_vector(
    values: List[float],
    source_joint_names: List[str],
    expected_joint_names: List[str],
    field_name: str,
    point_index: int,
) -> List[float]:
    if not values:
        return []

    if len(values) != len(source_joint_names):
        raise ValueError(
            f"Point {point_index}: field '{field_name}' length {len(values)} "
            f"does not match joint_names length {len(source_joint_names)}"
        )

    source_index: Dict[str, int] = {name: i for i, name in enumerate(source_joint_names)}
    return [float(values[source_index[j]]) for j in expected_joint_names]


def convert_joint_trajectory_to_yaml_dict(
    msg: DisplayTrajectory,
    expected_joint_names: List[str],
    allow_extra_joints: bool,
    topic_name: str,
) -> Dict[str, Any]:
    if not msg.trajectory:
        raise ValueError("DisplayTrajectory contains no RobotTrajectory entries.")

    selected = None
    selected_index = None

    for i, robot_traj in enumerate(msg.trajectory):
        jt = robot_traj.joint_trajectory
        if jt.points:
            selected = jt
            selected_index = i
            break

    if selected is None:
        raise ValueError("No non-empty joint_trajectory found in DisplayTrajectory.")

    source_joint_names = list(selected.joint_names)

    missing = [j for j in expected_joint_names if j not in source_joint_names]
    if missing:
        raise ValueError(
            "Captured trajectory does not contain all expected joints. "
            f"missing={missing}, source_joint_names={source_joint_names}"
        )

    extra = [j for j in source_joint_names if j not in expected_joint_names]
    if extra and not allow_extra_joints:
        raise ValueError(
            "Captured trajectory contains extra joints. "
            f"extra={extra}, expected={expected_joint_names}. "
            "If this is intentional, rerun with --allow-extra-joints."
        )

    if "Joint7" in source_joint_names:
        raise ValueError(
            "Captured trajectory contains Joint7. "
            "This exporter is for arm trajectory Joint1-Joint6 only."
        )

    points = []
    last_t = -1.0

    for idx, p in enumerate(selected.points):
        t = duration_to_sec(p.time_from_start)
        if t < last_t:
            raise ValueError(
                f"time_from_start is not monotonic at point {idx}: "
                f"{t} < previous {last_t}"
            )
        last_t = t

        positions = reorder_vector(
            list(p.positions),
            source_joint_names,
            expected_joint_names,
            "positions",
            idx,
        )

        if len(positions) != len(expected_joint_names):
            raise ValueError(
                f"Point {idx}: positions length {len(positions)} does not match "
                f"expected joints length {len(expected_joint_names)}"
            )

        velocities = reorder_vector(
            list(p.velocities),
            source_joint_names,
            expected_joint_names,
            "velocities",
            idx,
        )

        accelerations = reorder_vector(
            list(p.accelerations),
            source_joint_names,
            expected_joint_names,
            "accelerations",
            idx,
        )

        effort = reorder_vector(
            list(p.effort),
            source_joint_names,
            expected_joint_names,
            "effort",
            idx,
        )

        point_dict = {
            "time_from_start_sec": t,
            "positions": positions,
        }

        if velocities:
            point_dict["velocities"] = velocities
        if accelerations:
            point_dict["accelerations"] = accelerations
        if effort:
            point_dict["effort"] = effort

        points.append(point_dict)

    if not points:
        raise ValueError("Selected joint_trajectory has zero points.")

    return {
        "metadata": {
            "robot": "sukinee_urdf",
            "source": "moveit_display_planned_path",
            "topic": topic_name,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "display_trajectory_count": len(msg.trajectory),
            "selected_robot_trajectory_index": selected_index,
            "safety_boundary": {
                "socketcan": False,
                "type1_torque_ff": False,
                "type4_disable": False,
                "real_motor_command": False,
                "moveit_real_execution": False,
                "joint7_included": False,
            },
        },
        "trajectory": {
            "joint_names": expected_joint_names,
            "source_joint_names": source_joint_names,
            "point_count": len(points),
            "duration_sec": points[-1]["time_from_start_sec"],
            "points": points,
        },
    }


class DisplayTrajectoryCaptureNode(Node):
    def __init__(
        self,
        topic_name: str,
        output_yaml: str,
        expected_joint_names: List[str],
        allow_extra_joints: bool,
    ):
        super().__init__("sukinee_capture_display_trajectory")

        self.topic_name = topic_name
        self.output_yaml = output_yaml
        self.expected_joint_names = expected_joint_names
        self.allow_extra_joints = allow_extra_joints

        self.done = False
        self.failed = False
        self.error_message = ""

        self.sub = self.create_subscription(
            DisplayTrajectory,
            self.topic_name,
            self._on_display_trajectory,
            10,
        )

        self.get_logger().info(
            f"Waiting for MoveIt DisplayTrajectory on topic: {self.topic_name}"
        )
        self.get_logger().info(
            "In RViz MotionPlanning panel, click Plan for group sukinee_arm. "
            "Do NOT click Execute for this export step."
        )
        self.get_logger().info(f"Output YAML: {self.output_yaml}")
        self.get_logger().info(f"Expected joints: {self.expected_joint_names}")

    def _on_display_trajectory(self, msg: DisplayTrajectory) -> None:
        if self.done:
            return

        try:
            data = convert_joint_trajectory_to_yaml_dict(
                msg=msg,
                expected_joint_names=self.expected_joint_names,
                allow_extra_joints=self.allow_extra_joints,
                topic_name=self.topic_name,
            )

            output_dir = os.path.dirname(os.path.abspath(self.output_yaml))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(self.output_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )

            point_count = data["trajectory"]["point_count"]
            duration_sec = data["trajectory"]["duration_sec"]

            self.get_logger().info("Captured and exported trajectory successfully.")
            self.get_logger().info(f"points={point_count}, duration_sec={duration_sec:.6f}")
            self.get_logger().info(f"saved_to={self.output_yaml}")

            self.done = True

        except Exception as exc:
            self.failed = True
            self.done = True
            self.error_message = str(exc)
            self.get_logger().error(f"Failed to export trajectory: {self.error_message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture /display_planned_path and export Sukinee arm trajectory to YAML."
    )

    parser.add_argument(
        "--topic",
        default="/display_planned_path",
        help="DisplayTrajectory topic published by MoveIt/RViz.",
    )

    parser.add_argument(
        "--output-yaml",
        required=True,
        help="Output YAML path, e.g. /home/zzj/sukinee_ws/trajectories/plan_001.yaml",
    )

    parser.add_argument(
        "--expected-joints",
        nargs="+",
        default=DEFAULT_EXPECTED_JOINTS,
        help="Expected arm joints. Default: Joint1 Joint2 Joint3 Joint4 Joint5 Joint6",
    )

    parser.add_argument(
        "--allow-extra-joints",
        action="store_true",
        help="Allow captured trajectory to contain extra joints. Joint7 is still forbidden.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds while waiting for a DisplayTrajectory.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rclpy.init()

    node = DisplayTrajectoryCaptureNode(
        topic_name=args.topic,
        output_yaml=args.output_yaml,
        expected_joint_names=list(args.expected_joints),
        allow_extra_joints=bool(args.allow_extra_joints),
    )

    start_time = node.get_clock().now()

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

            elapsed = (node.get_clock().now() - start_time).nanoseconds * 1e-9
            if elapsed > args.timeout:
                node.failed = True
                node.done = True
                node.error_message = (
                    f"Timeout after {args.timeout:.1f} sec waiting for {args.topic}. "
                    "Make sure demo.launch.py is running and click Plan in RViz."
                )
                node.get_logger().error(node.error_message)
                break

        if node.failed:
            return 2

        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
