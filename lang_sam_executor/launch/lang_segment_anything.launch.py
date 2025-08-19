#!/usr/bin/env python3
"""
Launch file for LangSAM with Python CSRT Tracker (Fixed Label Mapping)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directories
    lang_sam_executor_dir = get_package_share_directory('lang_sam_executor')
    config_file = os.path.join(lang_sam_executor_dir, 'config', 'config.yaml')
    
    # Launch arguments
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode with verbose logging'
    )
    
    env_arg = DeclareLaunchArgument(
        'env',
        default_value='default',
        description='Environment configuration (default, production)'
    )
    
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=config_file,
        description='Path to configuration file'
    )
    
    # Python CSRT tracker node (uses fixed label mapping)
    lang_sam_tracker_node = Node(
        package='lang_sam_wrapper',
        executable='lang_sam_tracker_node.py',  # Using Python version with fixed tracking
        name='lang_sam_tracker_node',
        parameters=[LaunchConfiguration('config')],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info'] if not LaunchConfiguration('debug') else ['--ros-args', '--log-level', 'debug'],
        respawn=True,
        respawn_delay=5.0
    )
    
    # Multi-view visualization node
    multi_view_node = Node(
        package='lang_sam_wrapper',
        executable='multi_view_node.py',
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
        env_arg,
        config_arg,
        
        # Launch Python CSRT tracker node (fixed labels)
        lang_sam_tracker_node,
        
        # Launch multi-view visualization
        multi_view_node,
        
        # Log launch information
        ExecuteProcess(
            cmd=['echo', 'LangSAM Python CSRT Tracker launched successfully (Fixed Label Mapping)'],
            output='screen'
        )
    ])