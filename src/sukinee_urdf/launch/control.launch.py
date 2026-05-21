import os

import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'sukinee_urdf'

    package_share_path = get_package_share_directory(package_name)

    xacro_file_path = os.path.join(
        package_share_path,
        'urdf',
        'sukinee_urdf.urdf.xacro'
    )

    controllers_file_path = os.path.join(
        package_share_path,
        'config',
        'sukinee_controllers.yaml'
    )

    rviz_config_path = os.path.join(
        package_share_path,
        'rviz',
        'display.rviz'
    )

    robot_description_content = xacro.process_file(xacro_file_path).toxml()

    robot_description = {
        'robot_description': robot_description_content
    }

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[
            controllers_file_path
        ],
        remappings=[
            ('robot_description', '/robot_description')
        ]
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager'
        ],
        output='screen'
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_controller',
            '--controller-manager',
            '/controller_manager'
        ],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        rviz_node
    ])