#!/usr/bin/env python3
"""
Sukinee executor joint-state visualization launch.

Purpose:
  - Visualize /joint_states published by sukinee_type1_trajectory_executor.py.
  - Start robot_state_publisher.
  - Start a pure RViz RobotModel view.

Safety boundary:
  - Does NOT open can0.
  - Does NOT read motor feedback.
  - Does NOT send Type1.
  - Does NOT send Type3.
  - Does NOT send Type4.
  - Does NOT send Type6.
  - Does NOT send Type18.
  - Does NOT start move_group.
  - Does NOT start MoveIt MotionPlanning RViz plugin.
  - Does NOT start real feedback publisher.
  - Does NOT start zero_drag.
  - Does NOT execute MoveIt trajectories.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("sukinee_urdf", package_name="sukinee_moveit_config")
        .to_moveit_configs()
    )

    start_rviz = LaunchConfiguration("start_rviz")

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("sukinee_moveit_config"),
            "config",
            "sukinee_executor_joint_state_visualization.rviz",
        ]
    )

    declared_arguments = [
        DeclareLaunchArgument(
            "start_rviz",
            default_value="true",
            description="Start pure RViz RobotModel view to visualize executor-published /joint_states.",
        ),
    ]

    safety_banner = LogInfo(
        msg=[
            "\nSukinee executor joint-state visualization launch.\n",
            "This launch does not open can0 and does not send motor commands.\n",
            "It starts robot_state_publisher and a pure RViz RobotModel view only.\n",
            "It does not start move_group or MoveIt MotionPlanning plugin.\n",
            "Run executor with --publish-joint-states to animate the robot.\n",
        ]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description,
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_executor_joint_state_view",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        declared_arguments
        + [
            safety_banner,
            robot_state_publisher,
            rviz,
        ]
    )
