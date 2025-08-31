#!/usr/bin/env python3
"""
Launch file for LangSAM with Native C++ CSRT Tracker

技術的目的:
- ハイブリッドPython/C++システムの統合起動を管理する目的で使用
- メインAI/MLノードとマルチビュー可視化ノードの協調動作を実現
- デバッグモードと本番モードの柔軟な切り替えを提供する目的で実装
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # パッケージディレクトリ読み込み
    # 目的: config.yamlの絶対パスを解決しROS2パラメータとして読み込み
    lang_sam_executor_dir = get_package_share_directory('lang_sam_executor')
    config_file = os.path.join(lang_sam_executor_dir, 'config', 'config.yaml')
    
    # 起動引数定義
    # 目的: コマンドラインからデバッグモードや設定ファイルを動的に切り替え
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode with verbose logging'
    )
    
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file,
        description='Path to configuration file'
    )
    
    
    # メインLangSAMトラッカーノード（C++ CSRT統合）
    lang_sam_node = Node(
        package='lang_sam_wrapper',
        executable='lang_sam_tracker_node.py',
        name='lang_sam_tracker_node',
        parameters=[LaunchConfiguration('config')],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info'] if not LaunchConfiguration('debug') else ['--ros-args', '--log-level', 'debug'],
        respawn=True,
        respawn_delay=5.0
    )
    
    # マルチビュー可視化ノード（C++実装）
    multi_view_node = Node(
        package='lang_sam_executor',
        executable='multi_view_node',
        name='multi_view_node',
        parameters=[LaunchConfiguration('config')],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info'],
        respawn=True,
        respawn_delay=3.0
    )
    
    
    return LaunchDescription([
        debug_arg,
        config_arg,
        
        # メインLangSAMトラッカーノードを起動
        lang_sam_node,
        
        # マルチビュー可視化を起動
        multi_view_node,
        
        
        # 起動情報をログ出力
        ExecuteProcess(
            cmd=['echo', 'LangSAM Tracker launched successfully'],
            output='screen'
        )
    ])