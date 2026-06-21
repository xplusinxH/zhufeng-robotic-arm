#!/usr/bin/env python3
"""
Sukinee real-state MoveIt visualization launch.

Purpose:
  - Visualize the real Sukinee arm state in RViz using read-only motor feedback.
  - Start MoveIt move_group for plan-only visualization.
  - Start MoveIt RViz.
  - Optionally publish a target_tool0_preview TF frame.

Safety boundary:
  - The real feedback publisher uses Type17 read-only parameter reads.
  - This launch does NOT start zero_drag.
  - This launch does NOT start ros2_control real hardware.
  - This launch does NOT spawn trajectory controllers.
  - This launch does NOT send Type1 motion commands.
  - This launch does NOT send Type3 enable.
  - This launch does NOT send Type4 disable.
  - This launch does NOT send Type6 set zero.
  - This launch does NOT send Type18 write parameter.
  - This launch does NOT save motor parameters.
  - This launch does NOT change CAN_ID.
  - This launch does NOT switch protocol.
  - MoveIt is for planning / visualization only. Do NOT click Execute.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("sukinee_urdf", package_name="sukinee_moveit_config")
        .to_moveit_configs()
    )

    can_iface = LaunchConfiguration("can")
    offset_json = LaunchConfiguration("offset_json")
    feedback_rate = LaunchConfiguration("feedback_rate")
    feedback_print_every = LaunchConfiguration("feedback_print_every")

    start_move_group = LaunchConfiguration("start_move_group")
    start_rviz = LaunchConfiguration("start_rviz")
    start_target_preview = LaunchConfiguration("start_target_preview")

    target_x = LaunchConfiguration("target_x")
    target_y = LaunchConfiguration("target_y")
    target_z = LaunchConfiguration("target_z")
    target_roll = LaunchConfiguration("target_roll")
    target_pitch = LaunchConfiguration("target_pitch")
    target_yaw = LaunchConfiguration("target_yaw")

    declared_arguments = [
        DeclareLaunchArgument(
            "can",
            default_value="can0",
            description="SocketCAN interface for Type17 read-only feedback.",
        ),
        DeclareLaunchArgument(
            "offset_json",
            default_value=(
                "/home/zzj/sukinee_ws/vision_calibration/data/run_100_20260619_012357/"
                "gate5_result_77mm_exclude_P0021_P0008_joint2prior001/"
                "sukinee_motor_to_urdf_offsets_calibrated.json"
            ),
            description="Software-only motor_pos -> URDF q offset JSON.",
        ),
        DeclareLaunchArgument(
            "feedback_rate",
            default_value="1.0",
            description="Read-only /joint_states publish rate. Keep low for visualization.",
        ),
        DeclareLaunchArgument(
            "feedback_print_every",
            default_value="1",
            description="Print real q_urdf every N feedback cycles.",
        ),
        DeclareLaunchArgument(
            "start_move_group",
            default_value="true",
            description="Start MoveIt move_group for plan-only visualization.",
        ),
        DeclareLaunchArgument(
            "start_rviz",
            default_value="true",
            description="Start MoveIt RViz.",
        ),
        DeclareLaunchArgument(
            "start_target_preview",
            default_value="true",
            description="Publish target_tool0_preview TF.",
        ),
        DeclareLaunchArgument(
            "target_x",
            default_value="-0.192",
            description="Target tool0 preview x in base_link.",
        ),
        DeclareLaunchArgument(
            "target_y",
            default_value="-0.032",
            description="Target tool0 preview y in base_link.",
        ),
        DeclareLaunchArgument(
            "target_z",
            default_value="0.185",
            description="Target tool0 preview z in base_link.",
        ),
        DeclareLaunchArgument(
            "target_roll",
            default_value="-1.601",
            description="Target tool0 preview roll.",
        ),
        DeclareLaunchArgument(
            "target_pitch",
            default_value="-0.006",
            description="Target tool0 preview pitch.",
        ),
        DeclareLaunchArgument(
            "target_yaw",
            default_value="1.727",
            description="Target tool0 preview yaw.",
        ),
    ]

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description,
        ],
    )

    real_feedback_joint_state_publisher = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [
                    FindPackageShare("sukinee_urdf"),
                    "scripts",
                    "zero_drag_tools",
                    "sukinee_real_feedback_joint_state_publisher.py",
                ]
            ),
            "--can",
            can_iface,
            "--offset-json",
            offset_json,
            "--rate",
            feedback_rate,
            "--print-every",
            feedback_print_every,
        ],
        output="screen",
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("sukinee_moveit_config"),
                    "launch",
                    "move_group.launch.py",
                ]
            )
        ),
        condition=IfCondition(start_move_group),
    )

    moveit_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("sukinee_moveit_config"),
                    "launch",
                    "moveit_rviz.launch.py",
                ]
            )
        ),
        condition=IfCondition(start_rviz),
    )

    target_tool0_preview = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [
                    FindPackageShare("sukinee_moveit_config"),
                    "scripts",
                    "trajectory_exec",
                    "sukinee_preview_target_tool0_tf.py",
                ]
            ),
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "target_tool0_preview",
            "--target-x",
            target_x,
            "--target-y",
            target_y,
            "--target-z",
            target_z,
            "--target-roll",
            target_roll,
            "--target-pitch",
            target_pitch,
            "--target-yaw",
            target_yaw,
        ],
        output="screen",
        condition=IfCondition(start_target_preview),
    )

    safety_banner = LogInfo(
        msg=[
            "\nSukinee real-state MoveIt visualization launch.\n",
            "Safety boundary: Type17 read-only feedback + MoveIt plan visualization only.\n",
            "No zero_drag, no Type1 trajectory command, no Type3 enable, no Type4 disable, ",
            "no Type6 zero, no Type18 write, no parameter save, no CAN_ID/protocol change.\n",
            "Do NOT click Execute in RViz.\n",
        ]
    )

    return LaunchDescription(
        declared_arguments
        + [
            safety_banner,
            robot_state_publisher,
            real_feedback_joint_state_publisher,
            move_group,
            moveit_rviz,
            target_tool0_preview,
        ]
    )
