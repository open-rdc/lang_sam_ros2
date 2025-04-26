### lang_sam_ros2
---
#### lang-segment-anythingのインストール
https://github.com/luca-medeiros/lang-segment-anything

---
#### このリポジトリのインストール
```
mkdir -p ros2_ws/src
cd ros2_ws/src
git clone https://github.com/Ryusei-Baba/lang_sam_ros2.git
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```
---
#### 起動
```
ros2 launch lang_sam_ros2 lang_segment_anything.launch.py
```
---
#### License
This project is licensed under the Apache 2.0 License

