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
        super().__init__('lang_sam_mask_node')

        # ========================================
        # パラメータ宣言・取得
        # ========================================
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'car. wheel.')

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value

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
            ROSImage,
            '/image',
            self.image_callback,
            10
        )

        self.mask_pub = self.create_publisher(
            ROSImage,
            '/image_mask',
            10
        )

        self.get_logger().info("LangSAM Mask Node 起動完了")

    def image_callback(self, msg):
        try:
            # ========================================
            # ROS Image → OpenCV画像（RGB）
            # ========================================
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return

            cv_image = cv_image.astype(np.uint8, copy=True)
            image_pil = Image.fromarray(cv_image, mode='RGB')

            # ========================================
            # LangSAM 推論実行
            # ========================================
            start_time = time.time()
            results = self.model.predict([image_pil], [self.text_prompt])
            elapsed = time.time() - start_time
            self.get_logger().info(f"LangSAM推論時間: {elapsed:.3f} 秒")

        except Exception as e:
            self.get_logger().error(f"LangSAM推論エラー: {e}")
            return

        try:
            # ========================================
            # マスクを統合し、白黒画像として出力
            # ========================================
            combined_mask = (np.array(results[0]['masks']).sum(axis=0) > 0).astype(np.uint8) * 255

            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            self.mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"マスク描画・送信エラー: {e}")
