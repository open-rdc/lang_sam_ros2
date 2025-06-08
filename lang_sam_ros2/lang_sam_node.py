import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2

from PIL import Image
import numpy as np
import time

from lang_sam import LangSAM


class LangSAMNode(Node):
    def __init__(self):
        super().__init__('lang_sam_node')

        # ========================================
        # パラメータ宣言・取得
        # ========================================
        self.sam_model = self.declare_and_get_parameter('sam_model', 'sam2.1_hiera_small')
        self.text_prompt = self.declare_and_get_parameter('text_prompt', 'car. wheel.')

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # ========================================
        # モデルとユーティリティの初期化
        # ========================================
        self.model = LangSAM(sam_type=self.sam_model)
        self.bridge = CvBridge()

        # ========================================
        # ROS 2 通信設定（画像の購読とマスク画像の配信）
        # ========================================
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10)

        self.mask_pub = self.create_publisher(
            ROSImage, '/image_sam', 10)

        self.get_logger().info("LangSAM Mask Node 起動完了")

    def declare_and_get_parameter(self, name, default_value):
        """パラメータを宣言し、その値を取得する"""
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).get_parameter_value().string_value

    def image_callback(self, msg):
        try:
            # ROS Image → OpenCV画像（RGB）
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return

            cv_image = cv_image.astype(np.uint8, copy=True)
            image_pil = Image.fromarray(cv_image, mode='RGB')

            # LangSAM 推論実行
            start_time = time.time()
            results = self.model.predict([image_pil], [self.text_prompt])
            elapsed = time.time() - start_time
            # self.get_logger().info(f"LangSAM推論時間: {elapsed:.3f} 秒")

            # 結果のマスクを統合して送信
            self.publish_combined_mask(results)

        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {repr(e)}")

    def publish_combined_mask(self, results):
        """マスクを統合して送信する"""
        try:
            combined_mask = (np.array(results[0]['masks']).sum(axis=0) > 0).astype(np.uint8) * 255
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f"マスク描画・送信エラー: {repr(e)}")
