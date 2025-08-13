#!/usr/bin/env python3
"""
最適化されたLangSAM + オプティカルフロー統合launchファイル
テキストプロンプト → GroundingDINO → CSRT → SAM2セグメンテーション
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
        # LangSAM integrated tracking node
        Node(
            package='lang_sam_wrapper',
            executable='lang_sam_tracker_node',
            name='lang_sam_tracker_node',
            output='screen',
            parameters=[config_file],
            remappings=[
                ('/image', '/zed/zed_node/rgb/image_rect_color'),
                ('/image_sam', '/image_sam'),
                ('/sam_masks', '/sam_masks'),
            ],
        ),
        # マルチビュー統合ノード
        Node(
            package='lang_sam_wrapper',
            executable='multi_view_node',
            name='multi_view_node',
            output='screen',
            parameters=[config_file],
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