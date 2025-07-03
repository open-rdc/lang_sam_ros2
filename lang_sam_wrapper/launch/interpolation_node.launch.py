import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """interpolation_nodeのlaunchファイル"""
    
    # パッケージディレクトリ取得
    pkg_dir = get_package_share_directory('lang_sam_wrapper')
    
    # interpolation_nodeの設定
    interpolation_node = Node(
        package='lang_sam_wrapper',
        executable='interpolation_node',
        name='interpolation_node',
        parameters=[
            {'input_topic': '/sam_masks'},
            {'output_topic': '/sam_masks_interpolated'},
            {'image_topic': '/zed/zed_node/rgb/image_rect_color'},
            {'tracking_circle_radius': 3},
            {'tracking_circle_color': [0, 255, 0]},
            {'qos_size': 10}
        ],
        output='screen'
    )
    
    return LaunchDescription([
        interpolation_node,
    ])