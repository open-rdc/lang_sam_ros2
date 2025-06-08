from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # パッケージ内のconfig.yamlファイルのパスを取得
    config_file = os.path.join(
        FindPackageShare('lang_sam_ros2').find('lang_sam_ros2'),
        'config',
        'config.yaml'
    )

    # ノードの定義とLaunchDescriptionの生成
    return LaunchDescription([
        # Node(
        #     package='lang_sam_ros2',
        #     executable='debug_lang_sam_node', # デバッグ用のノード
        #     name='lang_sam_node',
        #     output='screen',
        #     parameters=[config_file],
        #     remappings=[
        #         ('/image', '/zed/zed_node/rgb/image_rect_color'),
        #         ('/image_sam', '/image_sam'),
        #     ],
        # ),
        Node(
            package='lang_sam_ros2',
            executable='main',  # メインのLangSAMノード
            output='screen',
            parameters=[config_file],
            remappings=[
                ('/image', '/zed/zed_node/rgb/image_rect_color'),
                ('/image_sam', '/image_sam'),
                ('/image_optflow', '/image_optflow'),
            ],
        ),
    ])