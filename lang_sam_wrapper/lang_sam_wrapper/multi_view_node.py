#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time
from collections import deque


class MultiViewNode(Node):
    """4分割画像表示（Original、GroundingDINO、CSRT、SAM2）+ 周波数監視"""
    
    def __init__(self):
        super().__init__('multi_view_node')
        
        # CvBridge（ROS Image ↔ OpenCV numpy変換）
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # パラメータ宣言・読み込み
        self._declare_all_parameters()
        self._load_all_parameters()
        
        # 4分割画像バッファ（BGR形式）
        self.images = {
            'original': None,      # 左上：ZED生画像
            'gdino': None,         # 右上：GroundingDINO検出結果
            'csrt': None,          # 左下：CSRT追跡結果
            'sam': None,           # 右下：SAM2セグメンテーション結果
        }
        
        # 周波数計算用タイムスタンプバッファ（5秒履歴、最大300フレーム@60Hz）
        self.timestamp_buffers = {
            'original': deque(maxlen=300),
            'gdino': deque(maxlen=300),
            'csrt': deque(maxlen=300),
            'sam': deque(maxlen=300),
        }
        
        # リアルタイム周波数（Hz）
        self.frequencies = {
            'original': 0.0,  # ZED画像更新頻度
            'gdino': 0.0,     # GroundingDINO実行頻度
            'csrt': 0.0,      # CSRTトラッキング更新頻度
            'sam': 0.0,       # SAM2セグメンテーション更新頻度
        }
        
        # サブスクライバー
        self.original_sub = self.create_subscription(
            Image, self.original_topic, self.original_callback, 10
        )
        self.gdino_sub = self.create_subscription(
            Image, self.gdino_topic, self.gdino_callback, 10
        )
        self.csrt_sub = self.create_subscription(
            Image, self.csrt_topic, self.csrt_callback, 10
        )
        self.sam_sub = self.create_subscription(
            Image, self.sam_topic, self.sam_callback, 10
        )
        
        # パブリッシャー
        self.multi_view_pub = self.create_publisher(Image, '/multi_view', 10)
        
        # 定期配信タイマー（指定FPSで統合画像配信）
        self.timer = self.create_timer(1.0 / self.output_fps, self.publish_multi_view)
        
        self.get_logger().info("MultiViewNode初期化完了（4分割+周波数監視）")
    
    def _declare_all_parameters(self):
        """全パラメータ宣言"""
        # トピック設定
        self.declare_parameter('original_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # 出力設定
        self.declare_parameter('output_fps', 30.0)
        self.declare_parameter('output_width', 1280)
        self.declare_parameter('output_height', 720)
        self.declare_parameter('border_width', 2)
        self.declare_parameter('border_color', [255, 255, 255])  # BGR
        
        # ラベル設定
        self.declare_parameter('enable_labels', True)
        self.declare_parameter('label_font_scale', 0.7)
        self.declare_parameter('label_thickness', 2)
        self.declare_parameter('label_color', [0, 255, 0])  # BGR
    
    def _load_all_parameters(self):
        """全パラメータ読み込み"""
        # トピック設定
        self.original_topic = self.get_parameter('original_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.csrt_topic = self.get_parameter('csrt_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
        
        # 出力設定
        self.output_fps = self.get_parameter('output_fps').value
        self.output_width = self.get_parameter('output_width').value
        self.output_height = self.get_parameter('output_height').value
        self.border_width = self.get_parameter('border_width').value
        self.border_color = tuple(self.get_parameter('border_color').value)
        
        # ラベル設定
        self.enable_labels = self.get_parameter('enable_labels').value
        self.label_font_scale = self.get_parameter('label_font_scale').value
        self.label_thickness = self.get_parameter('label_thickness').value
        self.label_color = tuple(self.get_parameter('label_color').value)
        
        # 個別画像サイズ計算
        self.single_width = (self.output_width - 3 * self.border_width) // 2
        self.single_height = (self.output_height - 3 * self.border_width) // 2
    
    def original_callback(self, msg: Image):
        """ZED生画像コールバック（周波数追跡付き）"""
        with self.lock:
            self.images['original'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._update_frequency('original')
    
    def gdino_callback(self, msg: Image):
        """GroundingDINO検出結果コールバック（周波数追跡付き）"""
        with self.lock:
            self.images['gdino'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._update_frequency('gdino')
    
    def csrt_callback(self, msg: Image):
        """CSRT追跡結果コールバック（周波数追跡付き）"""
        with self.lock:
            self.images['csrt'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._update_frequency('csrt')
    
    def sam_callback(self, msg: Image):
        """SAM2セグメンテーション結果コールバック（周波数追跡付き）"""
        with self.lock:
            self.images['sam'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._update_frequency('sam')
    
    def _update_frequency(self, topic_name: str):
        """指定トピック周波数計算（移動平均）"""
        current_time = time.time()
        self.timestamp_buffers[topic_name].append(current_time)
        
        # タイムスタンプバッファから周波数計算
        if len(self.timestamp_buffers[topic_name]) >= 2:
            time_span = current_time - self.timestamp_buffers[topic_name][0]
            if time_span > 0:
                self.frequencies[topic_name] = (len(self.timestamp_buffers[topic_name]) - 1) / time_span
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """画像リサイズ（4分割用、バイリニア補間）"""
        if image is None:
            # 黒画像生成（画像なしの場合）
            return np.zeros((self.single_height, self.single_width, 3), dtype=np.uint8)
        
        return cv2.resize(image, (self.single_width, self.single_height))
    
    def add_label(self, image: np.ndarray, text: str, position: str) -> np.ndarray:
        """OpenCVテキスト描画（周波数情報付きラベル）"""
        if not self.enable_labels:
            return image
        
        # FONT_HERSHEY_SIMPLEXでテキストサイズ計算
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                   self.label_font_scale, self.label_thickness)[0]
        
        # 位置別座標計算
        if position == 'top_left':
            x, y = 10, 30
        elif position == 'top_right':
            x, y = self.single_width - text_size[0] - 10, 30
        elif position == 'bottom_left':
            x, y = 10, self.single_height - 10
        else:  # bottom_right
            x, y = self.single_width - text_size[0] - 10, self.single_height - 10
        
        # OpenCVテキスト描画（BGR色空間）
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   self.label_font_scale, self.label_color, self.label_thickness)
        
        return image
    
    def create_multi_view(self) -> np.ndarray:
        """4分割統合画像作成（2x2グリッドレイアウト）"""
        with self.lock:
            # 各画像を統一サイズにリサイズ
            original = self.resize_image(self.images['original'])
            gdino = self.resize_image(self.images['gdino'])
            csrt = self.resize_image(self.images['csrt'])
            sam = self.resize_image(self.images['sam'])
        
        # ラベル追加（周波数情報付き）
        original = self.add_label(original, f"Original ({self.frequencies['original']:.1f}Hz)", 'top_left')
        gdino = self.add_label(gdino, f"GDINO ({self.frequencies['gdino']:.1f}Hz)", 'top_right')
        csrt = self.add_label(csrt, f"CSRT ({self.frequencies['csrt']:.1f}Hz)", 'bottom_left')
        sam = self.add_label(sam, f"SAM ({self.frequencies['sam']:.1f}Hz)", 'bottom_right')
        
        # BGR統合画像キャンバス作成（境界線付き）
        multi_view = np.full((self.output_height, self.output_width, 3), 
                           self.border_color, dtype=np.uint8)
        
        # 2x2グリッド配置座標
        positions = {
            'original': (self.border_width, self.border_width),  # 左上
            'gdino': (self.border_width + self.single_width + self.border_width, self.border_width),  # 右上
            'csrt': (self.border_width, self.border_width + self.single_height + self.border_width),  # 左下
            'sam': (self.border_width + self.single_width + self.border_width, 
                   self.border_width + self.single_height + self.border_width),  # 右下
        }
        
        # NumPy配列スライシングで画像配置
        for image_name, image in [('original', original), ('gdino', gdino), 
                                ('csrt', csrt), ('sam', sam)]:
            x, y = positions[image_name]
            multi_view[y:y + self.single_height, x:x + self.single_width] = image
        
        return multi_view
    
    def publish_multi_view(self):
        """4分割統合画像配信（ROS Image message）"""
        try:
            multi_view = self.create_multi_view()
            msg = self.bridge.cv2_to_imgmsg(multi_view, 'bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'multi_view'
            self.multi_view_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"MultiView配信エラー: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiViewNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()