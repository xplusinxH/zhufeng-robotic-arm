import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("sukinee_urdf")

    zero_torque_script = LaunchConfiguration("zero_torque_script")
    can = LaunchConfiguration("can")
    urdf = LaunchConfiguration("urdf")
    xacro_file = LaunchConfiguration("xacro")
    offset_json = LaunchConfiguration("offset_json")
    config_json = LaunchConfiguration("config_json")
    duration = LaunchConfiguration("duration")
    rate = LaunchConfiguration("rate")
    pos_timeout = LaunchConfiguration("pos_timeout")
    print_every = LaunchConfiguration("print_every")
    type1_delay = LaunchConfiguration("type1_delay")
    joint_state_every = LaunchConfiguration("joint_state_every")
    armed = LaunchConfiguration("armed")
    confirm = LaunchConfiguration("confirm")
    rviz_config = LaunchConfiguration("rviz_config")

    robot_description = {
        "robot_description": Command(["xacro ", xacro_file])
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    common_zero_drag_cmd = [
        "python3",
        zero_torque_script,
        "--can", can,
        "--urdf", urdf,
        "--offset-json", offset_json,
        "--config-json", config_json,
        "--duration", duration,
        "--rate", rate,
        "--pos-timeout", pos_timeout,
        "--print-every", print_every,
        "--type1-delay", type1_delay,
        "--joint-state-every", joint_state_every,
        "--publish-joint-states",
    ]

    zero_drag_dry_run = ExecuteProcess(
        cmd=common_zero_drag_cmd,
        output="screen",
        condition=UnlessCondition(armed),
    )

    zero_drag_armed = ExecuteProcess(
        cmd=common_zero_drag_cmd + [
            "--armed",
            "--confirm", confirm,
        ],
        output="screen",
        condition=IfCondition(armed),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "zero_torque_script",
            default_value="/home/zzj/sukinee_ws/src/sukinee_urdf/scripts/zero_drag/sukinee_zero_torque_gravity_mode_socketcan.py",
        ),
        DeclareLaunchArgument(
            "can",
            default_value="can0",
        ),
        DeclareLaunchArgument(
            "urdf",
            default_value="/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf",
        ),
        DeclareLaunchArgument(
            "xacro",
            default_value="/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf.xacro",
        ),
        DeclareLaunchArgument(
            "offset_json",
            default_value="/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json",
        ),
        DeclareLaunchArgument(
            "config_json",
            default_value="/home/zzj/sukinee_ws/sukinee_gravity_assist_config.json",
        ),
        DeclareLaunchArgument(
            "duration",
            default_value="0.0",
        ),
        DeclareLaunchArgument(
            "rate",
            default_value="0.0",
        ),
        DeclareLaunchArgument(
            "pos_timeout",
            default_value="0.03",
        ),
        DeclareLaunchArgument(
            "print_every",
            default_value="50",
        ),
        DeclareLaunchArgument(
            "type1_delay",
            default_value="0.0",
        ),
        DeclareLaunchArgument(
            "joint_state_every",
            default_value="2",
        ),
        DeclareLaunchArgument(
            "armed",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "confirm",
            default_value="I_UNDERSTAND_THIS_RUNS_SOCKETCAN_ZERO_TORQUE_GRAVITY_MODE",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([
                pkg_share,
                "rviz",
                "display.rviz",
            ]),
        ),

        robot_state_publisher_node,
        rviz_node,
        zero_drag_dry_run,
        zero_drag_armed,
    ])