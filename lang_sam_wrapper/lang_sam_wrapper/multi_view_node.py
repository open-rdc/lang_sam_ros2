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

# ROS2関係のユーティリティ
from lang_sam_wrapper.utils import MultiViewParameterManager, ImagePublisher, FrequencyMonitor


class MultiViewNode(Node):
    """4分割画像表示（Original、GroundingDINO、CSRT、SAM2）+ 周波数監視"""
    
    def __init__(self):
        super().__init__('multi_view_node')
        
        # 基本設定
        self.bridge = CvBridge()
        self.image_publisher = ImagePublisher(self)
        self.lock = threading.Lock()
        
        # パラメータ管理（簡素化）
        param_manager = MultiViewParameterManager(self)
        params = param_manager.initialize_parameters()
        self._load_parameters(params)
        
        # 4分割画像バッファ
        self.images = {
            'original': None, 'gdino': None, 'csrt': None, 'sam': None
        }
        
        # 周波数監視
        self.frequency_monitors = {
            'original': FrequencyMonitor(),
            'gdino': FrequencyMonitor(),
            'csrt': FrequencyMonitor(),
            'sam': FrequencyMonitor(),
        }
        
        # ROS2通信設定
        self._setup_communication()
        
        self.get_logger().info("MultiViewNode初期化完了")
    
    def _load_parameters(self, params: dict):
        """パラメータ読み込み"""
        self.original_topic = params['original_topic']
        self.gdino_topic = params['gdino_topic']
        self.csrt_topic = params['csrt_topic']
        self.sam_topic = params['sam_topic']
        self.output_fps = params['output_fps']
        self.output_width = params['output_width']
        self.output_height = params['output_height']
        self.border_width = params['border_width']
        self.border_color = tuple(params['border_color'])
        self.enable_labels = params['enable_labels']
        self.label_font_scale = params['label_font_scale']
        self.label_thickness = params['label_thickness']
        self.label_color = tuple(params['label_color'])
        
        # 計算値
        self.single_width = (self.output_width - 3 * self.border_width) // 2
        self.single_height = (self.output_height - 3 * self.border_width) // 2
    
    def _setup_communication(self):
        """ROS2通信設定"""
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
        """指定トピック周波数計算"""
        self.frequency_monitors[topic_name].update()
    
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
        original = self.add_label(original, f"Original ({self.frequency_monitors['original'].get_frequency():.1f}Hz)", 'top_left')
        gdino = self.add_label(gdino, f"GDINO ({self.frequency_monitors['gdino'].get_frequency():.1f}Hz)", 'top_right')
        csrt = self.add_label(csrt, f"CSRT ({self.frequency_monitors['csrt'].get_frequency():.1f}Hz)", 'bottom_left')
        sam = self.add_label(sam, f"SAM ({self.frequency_monitors['sam'].get_frequency():.1f}Hz)", 'bottom_right')
        
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