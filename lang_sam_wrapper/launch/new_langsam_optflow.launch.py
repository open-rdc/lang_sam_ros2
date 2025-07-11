#!/usr/bin/env python3
"""
新しいフロー用のlaunchファイル
テキストプロンプト → GroundingDINO → 特徴点抽出 → オプティカルフロー → SAM2セグメンテーション
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.substitutions import FindPackageShare


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
            executable='new_langsam_optflow_node',
            name='new_langsam_optflow_node',
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