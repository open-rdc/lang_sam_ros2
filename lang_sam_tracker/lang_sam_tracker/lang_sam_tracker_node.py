#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage

from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
import numpy as np

from lang_sam import LangSAM
from lang_sam.utils import draw_image
from lang_sam.models.utils import DEVICE

class LangSamTrackerNode(Node):
    def __init__(self):
        super().__init__('lang_sam_tracker')
        self.logger = self.get_logger()
        self.logger.info('Initializing LangSAM Tracker Node...')

        # パラメータの読み込み
        self._setup_parameters()

        # deviceの設定
        self.device = DEVICE

        # LangSAMモデルの初期化
        self.model = LangSAM()

        self.bridge = CvBridge()

        # ROS 2のサブスクライバとパブリッシャの設定
        self.image_sub = self.create_subscription(ROSImage, '/camera/image_raw', self.image_callback, 1)
        self.image_pub = self.create_publisher(ROSImage, '/lang_sam_output', 1)

        # ログの設定
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Using SAM model: {self.sam_model}')
        self.get_logger().info(f'Using text prompt: {self.text_prompt}')
        self.get_logger().info('LangSAM model initialized.')

    # パラメータ取得用のヘルパーメソッド
    def _setup_parameters(self):
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'wheel. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.25)

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value

    def image_callback(self, msg):
        # 画像メッセージをOpenCV形式に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # OpenCV画像をPIL画像に変換
        pil_image = PILImage.fromarray(cv_image)

        # LangSAMモデルで推論
        with torch.no_grad():
            results = self.model.predict([pil_image], [self.text_prompt])

        # 結果の描画
        result_image = draw_image(
            image_rgb=pil_image,
            masks=results[0]['masks'],
            xyxy=results[0]['boxes'],
            probs=results[0]['scores'],
            labels=results[0]['labels'],
        )

        # PIL画像をOpenCV形式に変換
        result_cv_image = np.array(result_image)

        # OpenCV画像をROS 2の画像メッセージに変換
        result_msg = self.bridge.cv2_to_imgmsg(result_cv_image, encoding='bgr8')

        # 結果のパブリッシュ
        self.image_pub.publish(result_msg)
        # self.get_logger().info('Published processed image.')

    

def main():
    rclpy.init()
    node = LangSamTrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
