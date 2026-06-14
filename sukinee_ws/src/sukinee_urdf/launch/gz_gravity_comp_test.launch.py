import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package_name = 'sukinee_urdf'
    package_share = get_package_share_directory(package_name)

    zero_gravity_world = os.path.join(
        package_share,
        'worlds',
        'sukinee_zero_gravity.sdf',
    )

    gz_effort_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, 'launch', 'gz_effort.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r ', zero_gravity_world],
        }.items(),
    )

    gravity_comp_script = os.path.join(
        package_share,
        'scripts',
        'gz_gravity_comp_node.py',
    )

    arm_effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_effort_controller',
            '--controller-manager',
            '/controller_manager',
        ],
        output='screen',
    )

    gravity_comp_node = ExecuteProcess(
        cmd=[
            'python3',
            gravity_comp_script,
            '--rate', '150.0',
            '--log-interval', '1.0',
            '--gravity-scales', '0.0', '1.0', '1.0', '1.0', '0.0', '0.0',
            '--damping', '0.05', '0.25', '0.28', '2.0', '1.20', '1.20',
            '--joint-signs', '1', '1', '1', '1', '1', '1',
            '--effort-limits', '0.2', '2.0', '2.0', '1.2', '0.22', '0.22',
            '--torque-rate-limits', '1.0', '40.0', '30.0', '200.0', '1.5', '1.5',
            '--startup-ramp-time', '0.0',
            '--state-timeout', '0.6',
            '--max-velocity', '3.30',
        ],
        output='screen',
    )

    restore_gravity = ExecuteProcess(
        cmd=[
            'gz', 'service',
            '-s', '/world/empty/set_physics',
            '--reqtype', 'gz.msgs.Physics',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', 'gravity { x: 0 y: 0 z: -9.8 } enable_physics: true max_step_size: 0.001 real_time_factor: 1.0',
        ],
        output='screen',
    )

    return LaunchDescription([
        gz_effort_launch,
        TimerAction(
            period=5.0,
            actions=[arm_effort_controller_spawner],
        ),
        TimerAction(
            period=6.0,
            actions=[gravity_comp_node],
        ),
        TimerAction(
            period=6.3,
            actions=[restore_gravity],
        ),
    ])
