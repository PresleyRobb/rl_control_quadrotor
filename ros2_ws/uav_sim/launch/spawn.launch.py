from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

WORLD = os.path.expanduser('~/gazebo_models/worlds/crazyflie_world.sdf')

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['gz', 'sim', '--verbose', '-r', WORLD],
            output='screen'),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                # ROS → Gazebo (works)
                '/crazyflie/gazebo/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist',
                # Gazebo → ROS  (fixed)
                '/model/crazyflie/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                # Camera
                '/camera@sensor_msgs/msg/Image@gz.msgs.Image'
            ],
            output='screen'),
    ])

