
import argparse
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive


EXPECTED_ARM_JOINTS = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]


def rpy_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert roll/pitch/yaw to geometry quaternion values."""
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

    source_index = {name: i for i, name in enumerate(source_joint_names)}
    return [float(values[source_index[j]]) for j in expected_joint_names]


def make_pose(
    frame_id: str,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> PoseStamped:
    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

    ps = PoseStamped()
    ps.header.frame_id = frame_id

    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)

    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw

    return ps


def make_pose_goal_constraints(
    target_pose: PoseStamped,
    eef_link: str,
    position_tolerance: float,
    orientation_tolerance: float,
) -> Constraints:
    constraints = Constraints()
    constraints.name = f"{eef_link}_pose_goal"

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [
        float(position_tolerance) * 2.0,
        float(position_tolerance) * 2.0,
        float(position_tolerance) * 2.0,
    ]

    region_pose = Pose()
    region_pose.position = target_pose.pose.position
    region_pose.orientation.w = 1.0

    pc = PositionConstraint()
    pc.header.frame_id = target_pose.header.frame_id
    pc.link_name = eef_link
    pc.target_point_offset.x = 0.0
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0
    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(region_pose)
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = target_pose.header.frame_id
    oc.link_name = eef_link
    oc.orientation = target_pose.pose.orientation
    oc.absolute_x_axis_tolerance = float(orientation_tolerance)
    oc.absolute_y_axis_tolerance = float(orientation_tolerance)
    oc.absolute_z_axis_tolerance = float(orientation_tolerance)
    oc.weight = 1.0

    constraints.position_constraints.append(pc)
    constraints.orientation_constraints.append(oc)

    return constraints


def build_move_group_goal(args) -> MoveGroup.Goal:
    target_pose = make_pose(
        frame_id=args.frame_id,
        x=args.target_x,
        y=args.target_y,
        z=args.target_z,
        roll=args.target_roll,
        pitch=args.target_pitch,
        yaw=args.target_yaw,
    )

    request = MotionPlanRequest()
    request.group_name = args.group
    request.num_planning_attempts = int(args.num_planning_attempts)
    request.allowed_planning_time = float(args.allowed_planning_time)
    request.max_velocity_scaling_factor = float(args.max_velocity_scaling)
    request.max_acceleration_scaling_factor = float(args.max_acceleration_scaling)

    if args.planner_id:
        request.planner_id = args.planner_id

    # Empty start_state means MoveIt uses current PlanningScene start state.
    request.start_state.is_diff = True

    request.goal_constraints.append(
        make_pose_goal_constraints(
            target_pose=target_pose,
            eef_link=args.eef_link,
            position_tolerance=args.position_tolerance,
            orientation_tolerance=args.orientation_tolerance,
        )
    )

    options = PlanningOptions()
    options.plan_only = True
    options.look_around = False
    options.replan = False

    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options = options

    return goal


def convert_robot_trajectory_to_yaml_dict(
    robot_trajectory,
    expected_joint_names: List[str],
    target_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    jt = robot_trajectory.joint_trajectory

    if not jt.points:
        raise ValueError("MoveIt result contains an empty joint_trajectory.")

    source_joint_names = list(jt.joint_names)

    missing = [j for j in expected_joint_names if j not in source_joint_names]
    if missing:
        raise ValueError(
            "Planned trajectory does not contain all expected arm joints. "
            f"missing={missing}, source_joint_names={source_joint_names}"
        )

    extra = [j for j in source_joint_names if j not in expected_joint_names]
    if extra:
        raise ValueError(
            "Planned trajectory contains extra joints. "
            f"extra={extra}, expected={expected_joint_names}"
        )

    if "Joint7" in source_joint_names:
        raise ValueError("Planned arm trajectory contains Joint7, which is forbidden.")

    points = []
    last_t = -1.0

    for idx, p in enumerate(jt.points):
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

    return {
        "metadata": {
            "robot": "sukinee_urdf",
            "source": "moveit_move_action_plan_only",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": target_metadata,
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


class MoveItPosePlanExportNode(Node):
    def __init__(self, args):
        super().__init__("sukinee_plan_to_pose_export")
        self.args = args
        self.action_client = ActionClient(self, MoveGroup, args.action_name)

    def plan(self) -> Any:
        self.get_logger().info(f"Waiting for MoveGroup action server: {self.args.action_name}")
        if not self.action_client.wait_for_server(timeout_sec=self.args.server_timeout):
            raise RuntimeError(
                f"MoveGroup action server '{self.args.action_name}' not available. "
                "Make sure demo.launch.py is running."
            )

        goal = build_move_group_goal(self.args)

        self.get_logger().info("Sending MoveIt plan_only request.")
        self.get_logger().info(
            f"group={self.args.group}, eef_link={self.args.eef_link}, frame_id={self.args.frame_id}"
        )
        self.get_logger().info(
            f"target xyz=({self.args.target_x:.6f}, {self.args.target_y:.6f}, {self.args.target_z:.6f}), "
            f"rpy=({self.args.target_roll:.6f}, {self.args.target_pitch:.6f}, {self.args.target_yaw:.6f})"
        )

        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self.args.goal_response_timeout)

        goal_handle = send_future.result()
        if goal_handle is None:
            raise RuntimeError("MoveGroup goal response future returned None.")

        if not goal_handle.accepted:
            raise RuntimeError("MoveGroup action goal was rejected.")

        self.get_logger().info("MoveGroup goal accepted. Waiting for planning result.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.args.result_timeout)

        if not result_future.done():
            raise RuntimeError("Timeout waiting for MoveGroup planning result.")

        result_wrapper = result_future.result()
        if result_wrapper is None:
            raise RuntimeError("MoveGroup result future returned None.")

        result = result_wrapper.result
        error_code = int(result.error_code.val)

        if error_code != 1:
            raise RuntimeError(
                f"MoveIt planning failed. error_code={error_code}. "
                "Check RViz/move_group logs for details."
            )

        self.get_logger().info(
            f"MoveIt planning succeeded. planning_time={result.planning_time:.6f} sec"
        )

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan tool0 pose target through MoveIt move_action and export trajectory YAML."
    )

    parser.add_argument("--action-name", default="/move_action")
    parser.add_argument("--group", default="sukinee_arm")
    parser.add_argument("--eef-link", default="tool0")
    parser.add_argument("--frame-id", default="base_link")

    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-z", type=float, required=True)
    parser.add_argument("--target-roll", type=float, required=True)
    parser.add_argument("--target-pitch", type=float, required=True)
    parser.add_argument("--target-yaw", type=float, required=True)

    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.05)

    parser.add_argument("--planner-id", default="")
    parser.add_argument("--num-planning-attempts", type=int, default=5)
    parser.add_argument("--allowed-planning-time", type=float, default=5.0)

    parser.add_argument("--max-velocity-scaling", type=float, default=0.2)
    parser.add_argument("--max-acceleration-scaling", type=float, default=0.2)

    parser.add_argument("--server-timeout", type=float, default=10.0)
    parser.add_argument("--goal-response-timeout", type=float, default=10.0)
    parser.add_argument("--result-timeout", type=float, default=30.0)

    parser.add_argument("--output-yaml", required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rclpy.init()
    node = MoveItPosePlanExportNode(args)

    try:
        result = node.plan()

        target_metadata = {
            "frame_id": args.frame_id,
            "eef_link": args.eef_link,
            "group": args.group,
            "x": args.target_x,
            "y": args.target_y,
            "z": args.target_z,
            "roll": args.target_roll,
            "pitch": args.target_pitch,
            "yaw": args.target_yaw,
            "position_tolerance": args.position_tolerance,
            "orientation_tolerance": args.orientation_tolerance,
            "planner_id": args.planner_id,
            "num_planning_attempts": args.num_planning_attempts,
            "allowed_planning_time": args.allowed_planning_time,
            "max_velocity_scaling": args.max_velocity_scaling,
            "max_acceleration_scaling": args.max_acceleration_scaling,
        }

        data = convert_robot_trajectory_to_yaml_dict(
            robot_trajectory=result.planned_trajectory,
            expected_joint_names=EXPECTED_ARM_JOINTS,
            target_metadata=target_metadata,
        )

        output_yaml = os.path.abspath(args.output_yaml)
        os.makedirs(os.path.dirname(output_yaml), exist_ok=True)

        with open(output_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

        node.get_logger().info("Exported planned trajectory successfully.")
        node.get_logger().info(
            f"points={data['trajectory']['point_count']}, "
            f"duration_sec={data['trajectory']['duration_sec']:.6f}"
        )
        node.get_logger().info(f"saved_to={output_yaml}")

        return 0

    except Exception as exc:
        node.get_logger().error(str(exc))
        return 2

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
