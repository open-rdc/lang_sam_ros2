### lang_sam_ros2

本手法は，Language Segment-Anything にKLTトラッカー（Kanade-Lucas-Tomasi Feature Tracker）を導入することで，ゼロショットで高速なトラッキングを実現します．ROS 2環境での利用に対応しています．

[[Paper](doc/si2025_main1.pdf)]

---
#### 概要（後日修正予定）
下図左は LangSAM の出力，右はそのマスクをKLTトラッカーでトラッキングした出力です．

<div align="center">
  <table>
    <tr>
      <td><img src="./doc/lang_sam_mask.gif" alt="LangSAM 出力" height="200"></td>
      <td><img src="./doc/optical_flow.gif" alt="Optical Flow 出力" height="200"></td>
    </tr>
    <tr>
      <td align="center"><b>LangSAMによる検出<br>text_prompt: "white line. human. red pylon. wall. car. building. mobility. road."</b></td>
      <td align="center"><b>KLTトラッカーによる追跡<br>tracking_targets: "white line. human. red pylon. car. mobility."</b></td>
    </tr>
  </table>
</div>


---
#### lang-segment-anythingのインストール
https://github.com/luca-medeiros/lang-segment-anything

---
#### このリポジトリのインストール
```bash
mkdir -p ros2_ws/src
cd ros2_ws/src
git clone https://github.com/open-rdc/lang_sam_ros2.git
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```
---
#### 起動
```bash
ros2 launch lang_sam_executor lang_segment_anything.launch.py
```

---
#### License
This project is licensed under the Apache 2.0 License

