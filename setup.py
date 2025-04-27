import glob
from setuptools import find_packages, setup

package_name = 'lang_sam_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
    ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'cv_bridge',
        'lang_sam',
    ],
    zip_safe=True,
    maintainer='Ryusei Baba',
    maintainer_email='babaryusei.kw@gmail.com',
    description='LangSAMを使ってROS2でマスク生成するノード',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lang_segment_anything = lang_sam_ros2.lang_segment_anything:main'
        ],
    },
)
