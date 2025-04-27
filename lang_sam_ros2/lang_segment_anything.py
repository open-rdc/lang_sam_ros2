import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from PIL import Image
import numpy as np
import time

from lang_sam import LangSAM
from lang_sam.utils import draw_image
class LangSAMNode(Node):
    def __init__(self):
        super().__init__('lang_sam_node')

        # --- パラメータ宣言 ---
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'car. wheel.')

        # --- パラメータ取得 ---
        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # --- LangSAMセットアップ ---
        self.model = LangSAM(sam_type=self.sam_model)
        self.bridge = CvBridge()

        # --- ROS2通信設定 ---
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

        self.get_logger().info("LangSAMNode 起動完了")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge変換失敗: {e}")
            return

        image_pil = Image.fromarray(cv_image)

        # --- 推論時間計測スタート ---
        start_time = time.time()

        results = self.model.predict([image_pil], [self.text_prompt])

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.get_logger().info(f"LangSAM推論時間: {elapsed_time:.3f}秒")
        # --- 推論時間計測ここまで ---

        masks = results[0]['masks']
        boxes = results[0]['boxes']
        scores = results[0]['scores']
        labels = results[0]['labels']

        annotated_image = draw_image(
            image_rgb=cv_image,
            masks=np.array(masks),
            xyxy=np.array(boxes),
            probs=np.array(scores),
            labels=labels
        )

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge変換失敗: {e}")
            return

        self.mask_pub.publish(mask_msg)


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