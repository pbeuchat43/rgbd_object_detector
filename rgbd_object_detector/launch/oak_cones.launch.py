"""Explicit development composition; installing this file starts no workload."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('device_id', default_value=''),
        DeclareLaunchArgument('engine_path'),
        DeclareLaunchArgument('namespace', default_value='perception'),
        DeclareLaunchArgument('enable_imu', default_value='false'),
        ComposableNodeContainer(
            name='perception_container', namespace=LaunchConfiguration('namespace'),
            package='rclcpp_components', executable='component_container_mt', output='screen',
            composable_node_descriptions=[
                ComposableNode(
                    package='rgbd_camera', plugin='rgbd_camera::CameraNode', name='camera',
                    namespace=LaunchConfiguration('namespace'),
                    parameters=[{
                        'device_id': ParameterValue(LaunchConfiguration('device_id'), value_type=str),
                        'enable_imu': ParameterValue(LaunchConfiguration('enable_imu'), value_type=bool),
                    }], extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='rgbd_object_detector', plugin='rgbd_object_detector::DetectorNode', name='detector',
                    namespace=LaunchConfiguration('namespace'),
                    parameters=[{'engine_path': ParameterValue(LaunchConfiguration('engine_path'), value_type=str)}],
                    remappings=[('frames', 'camera/frames')],
                    extra_arguments=[{'use_intra_process_comms': True}]),
            ]),
    ])
