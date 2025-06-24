from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # パッケージ内のconfig.yamlファイルのパスを取得
    config_file = os.path.join(
        FindPackageShare('lang_sam_wrapper').find('lang_sam_wrapper'),
        'config',
        'config.yaml'
    )

    # ノードの定義とLaunchDescriptionの生成
    return LaunchDescription([
        # 統合されたLangSAM + Optical Flowノード
        Node(
            package='lang_sam_wrapper',
            executable='optflow_node',
            name='optflow_node',
            output='screen',
            parameters=[config_file],
            remappings=[
                ('/image', '/zed/zed_node/rgb/image_rect_color'),
                ('/image_sam', '/image_sam'),
                ('/sam_masks', '/sam_masks'),
                ('/image_optflow', '/image_optflow'),
            ],
        ),
        # デバッグ用LangSAMノード（必要に応じてコメントアウト解除）
        # Node(
        #     package='lang_sam_wrapper',
        #     executable='debug_lang_sam_node',
        #     name='debug_lang_sam_node',
        #     output='screen',
        #     parameters=[config_file],
        #     remappings=[
        #         ('/image', '/zed/zed_node/rgb/image_rect_color'),
        #         ('/image_sam', '/image_sam_debug'),
        #     ],
        # ),
    ])