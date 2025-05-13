from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    config_file = os.path.join(
        FindPackageShare('lang_sam_ros2').find('lang_sam_ros2'),
        'config',
        'config.yaml'
    )

    return LaunchDescription([
        Node(
            package='lang_sam_ros2',
            executable='lang_segment_anything',
            name='lang_sam_node',
            output='screen',
            parameters=[config_file],
            remappings=[
                ('/image', '/zed_node/rgb/image_rect_color'),
                ('/image_mask', '/image_mask')
            ],
        ),
    ])
