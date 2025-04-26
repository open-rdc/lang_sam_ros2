from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        # パラメータをコマンドライン引数で受け取れるようにする
        DeclareLaunchArgument(
            'sam_model',
            default_value='sam2.1_hiera_tiny',
            description='SAM model type to use'
        ),
        DeclareLaunchArgument(
            'text_prompt',
            default_value='white line. human.',
            description='Text prompt for LangSAM'
        ),

        Node(
            package='lang_sam_ros2',
            executable='lang_segment_anything',
            name='lang_segment_anything',
            output='screen',
            parameters=[{
                'sam_model': LaunchConfiguration('sam_model'),
                'text_prompt': LaunchConfiguration('text_prompt'),
            }],
            remappings=[
                ('/image', '/zed/zed_node/rgb/image_rect_color'), # 入力変換
                ('/image_mask', '/image_mask') # 出力変換
            ],
        ),
    ])
