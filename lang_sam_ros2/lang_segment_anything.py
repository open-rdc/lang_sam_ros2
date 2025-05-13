import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2

from PIL import Image
import numpy as np
import time

from lang_sam import LangSAM
from lang_sam.utils import draw_image


class LangSAMNode(Node):
    def __init__(self):
        super().__init__('lang_sam_node')

        # -------------------------------
        # パラメータ宣言・取得
        # -------------------------------
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'car. wheel.')

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # -------------------------------
        # LangSAM モデル・ツール初期化
        # -------------------------------
        self.model = LangSAM(sam_type=self.sam_model)
        self.bridge = CvBridge()

        # -------------------------------
        # ROS 2 通信設定（購読・配信）
        # -------------------------------
        self.image_sub = self.create_subscription(
            ROSImage,
            '/image',              # 入力画像トピック
            self.image_callback,
            10
        )

        self.mask_pub = self.create_publisher(
            ROSImage,
            '/image_mask',         # セグメンテーション結果トピック
            10
        )

        self.get_logger().info("LangSAMNode 起動完了")

    def image_callback(self, msg):
        try:
            # -------------------------------
            # 受信画像をNumPy形式に変換
            # -------------------------------
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image = np.asarray(cv_image)

            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return

            # uint8型かつ書き込み可能な形式に明示的に変換（Jetson環境の警告対策）
            cv_image = cv_image.astype(np.uint8, copy=True)

            # RGBA画像 → RGB → PIL形式に変換
            image_pil = Image.fromarray(cv_image, mode='RGBA').convert('RGB')

            # -------------------------------
            # LangSAM 推論実行
            # -------------------------------
            start_time = time.time()
            results = self.model.predict([image_pil], [self.text_prompt])
            elapsed = time.time() - start_time
            # self.get_logger().info(f"LangSAM推論時間: {elapsed:.3f} 秒")

        except Exception as e:
            self.get_logger().error(f"LangSAM推論エラー: {e}")
            return

        try:
            # -------------------------------
            # セグメンテーション結果を描画
            # -------------------------------
            annotated_image = draw_image(
                image_rgb=np.array(image_pil, copy=True),
                masks=np.array(results[0]['masks']),
                xyxy=np.array(results[0]['boxes']),
                probs=np.array(results[0]['scores']),
                labels=results[0]['labels']
            )

            # 結果画像をROSメッセージに変換し配信
            mask_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            self.mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"マスク描画・送信エラー: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LangSAMNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
