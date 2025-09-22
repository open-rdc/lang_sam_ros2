#!/usr/bin/env python3
"""
Navigation Launch file for Lane Following and Multi-view
Separated from main LangSAM tracking system
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # パッケージディレクトリ
    lang_sam_executor_dir = get_package_share_directory('lang_sam_executor')
    config_file = os.path.join(lang_sam_executor_dir, 'config', 'config.yaml')
    
    # 起動引数
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
    
    enable_nav_arg = DeclareLaunchArgument(
        'enable_navigation',
        default_value='true',
        description='Enable lane following navigation node'
    )
    
    enable_multiview_arg = DeclareLaunchArgument(
        'enable_multiview',
        default_value='true',
        description='Enable multi-view visualization node'
    )
    
    # 車線追従ナビゲーションノード（条件付き）
    lane_following_node = Node(
        package='lang_sam_nav',
        executable='lane_following_node',
        name='lane_following_node',
        parameters=[LaunchConfiguration('config')],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info'],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(LaunchConfiguration('enable_navigation'))
    )
    
    return LaunchDescription([
        debug_arg,
        config_arg,
        enable_nav_arg,
        enable_multiview_arg,
        
        # 車線追従ナビゲーションを起動（条件付き）
        lane_following_node,
        
        # 起動情報をログ出力
        ExecuteProcess(
            cmd=['echo', 'Navigation system (Multi-view + Lane Following) launched successfully'],
            output='screen'
        )
    ])