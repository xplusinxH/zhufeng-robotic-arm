import os

import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'sukinee_urdf'
    package_share = get_package_share_directory(package_name)
    gazebo_resource_path = os.path.dirname(package_share)
    xacro_file = os.path.join(
        package_share,
        'urdf',
        'sukinee_urdf.urdf.xacro'
    )

    controllers_file = os.path.join(
        package_share,
        'config',
        'sukinee_gz_effort_controllers.yaml'
    )

    robot_description_content = xacro.process_file(
        xacro_file,
        mappings={
            'use_mock_hardware': 'false',
            'hardware_plugin': 'gz_ros2_control/GazeboSimSystem',
            'arm_command_interface': 'effort',
            'gripper_command_interface': 'position',
            'include_gz_plugin': 'true',
            'controllers_file': controllers_file,
            'world_z': '0.30',
            'use_gazebo_gripper_controller': 'true',
        }
    ).toxml()

    robot_description = {
        'robot_description': robot_description_content
    }

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': LaunchConfiguration('gz_args')
        }.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'sukinee',
            '-topic', 'robot_description',
            '-allow_renaming', 'true',
        ]
    )
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        parameters=[
            {'bridge_names': ['clock_bridge']},
            {'bridges.clock_bridge.ros_topic_name': '/clock'},
            {'bridges.clock_bridge.gz_topic_name': '/clock'},
            {'bridges.clock_bridge.ros_type_name': 'rosgraph_msgs/msg/Clock'},
            {'bridges.clock_bridge.gz_type_name': 'gz.msgs.Clock'},
            {'bridges.clock_bridge.direction': 'GZ_TO_ROS'},
            {'bridges.clock_bridge.lazy': False},
            {'bridges.clock_bridge.qos_profile': 'CLOCK'},
        ],
    )
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
        ],
        output='screen'
    )

    arm_effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_effort_controller',
            '--controller-manager',
            '/controller_manager',
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gz_args',
            default_value='-r empty.sdf',
            description='Arguments passed to gz sim. Use "empty.sdf" to start paused.',
        ),
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=gazebo_resource_path + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        ),
        SetEnvironmentVariable(
            name='IGN_GAZEBO_RESOURCE_PATH',
            value=gazebo_resource_path + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
        ),
        gz_sim,
        clock_bridge,
        robot_state_publisher,
        spawn_robot,
        TimerAction(
            period=4.0,
            actions=[
                joint_state_broadcaster_spawner,
                # arm_effort_controller_spawner,
            ]
        ),
    ])