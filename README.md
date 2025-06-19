### lang_sam_ros2

本パッケージは，**Language Segment-Anything** を ROS 2 で使用できるようにしたものです．

---
#### 概要
**Language Segment-Anything** と **オプティカルフロー処理** を統合することで， 
<strong>ゼロショットで高速なトラッキング</strong> を実現しています．

下図左は LangSAM のマスク出力，右はそのマスクをオプティカルフローでトラッキングした結果です．

<div align="center">
  <table>
    <tr>
      <td><img src="./doc/lang_sam_mask.gif" alt="LangSAM 出力" height="200"></td>
      <td><img src="./doc/optical_flow.gif" alt="Optical Flow 出力" height="200"></td>
    </tr>
    <tr>
      <td align="center"><b>LangSAM 出力<br>text_prompt: "white line. human. red pylon. wall. car. building. mobility. road."</b></td>
      <td align="center"><b>オプティカルフローによる追跡<br>tracking_targets: "white line. human. red pylon. car. mobility."</b></td>
    </tr>
  </table>
</div>


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
ros2 launch lang_sam_wrapper lang_segment_anything.launch.py
```
---
#### License
This project is licensed under the Apache 2.0 License

