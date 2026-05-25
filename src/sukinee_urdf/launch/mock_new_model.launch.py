from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_file = LaunchConfiguration("urdf_file")
    controllers_file = LaunchConfiguration("controllers_file")

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            urdf_file,
            " ",
            "use_mock_hardware:=true",
            " ",
            "hardware_plugin:=mock_components/GenericSystem",
            " ",
            "arm_command_interface:=position",
            " ",
            "gripper_command_interface:=position",
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        )
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "urdf_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sukinee_urdf"),
                        "urdf",
                        "sukinee_urdf_new_model.urdf.xacro",
                    ]
                ),
                description="Path to the new Sukinee URDF/Xacro file.",
            ),

            DeclareLaunchArgument(
                "controllers_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sukinee_urdf"),
                        "config",
                        "sukinee_controllers.yaml",
                    ]
                ),
                description="Path to ros2_control controllers YAML.",
            ),

            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[robot_description],
            ),

            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[
                    robot_description,
                    controllers_file,
                ],
                output="screen",
            ),

            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "joint_state_broadcaster",
                    "--controller-manager",
                    "/controller_manager",
                ],
                output="screen",
            ),

            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "arm_controller",
                    "--controller-manager",
                    "/controller_manager",
                ],
                output="screen",
            ),

            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "gripper_controller",
                    "--controller-manager",
                    "/controller_manager",
                ],
                output="screen",
            ),

            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
            ),
        ]
    )