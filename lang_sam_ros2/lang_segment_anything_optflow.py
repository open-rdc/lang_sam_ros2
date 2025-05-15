import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from PIL import Image
import numpy as np
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from lang_sam import LangSAM
from lang_sam.utils import draw_image


class LangSAMNode(Node):
    def __init__(self):
        super().__init__('lang_sam_node')

        # =====================================
        # パラメータの宣言と取得
        # =====================================
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'car. wheel.')

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # =====================================
        # モデル、ツール、スレッド、同期
        # =====================================
        self.model = LangSAM(sam_type=self.sam_model)
        self.bridge = CvBridge()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.predicting = False
        self.lock = threading.Lock()

        # =====================================
        # ROS通信設定（サブスクライバ・パブリッシャ）
        # =====================================
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10
        )
        self.mask_pub = self.create_publisher(
            ROSImage, '/image_mask', 10
        )
        self.flow_pub = self.create_publisher(
            ROSImage, '/flow_debug', 10
        )

        # =====================================
        # 前フレームの情報保持（補間用）
        # =====================================
        self.prev_image = None
        self.prev_masks = None
        self.prev_labels = None
        self.prev_boxes = None
        self.prev_scores = None

        self.get_logger().info("LangSAMNode（マスク補間あり）起動完了")

    def image_callback(self, msg):
        try:
            # =====================================
            # ROS Image → OpenCV画像（RGB）
            # =====================================
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return

            # 書き込み可能な形式へ変換
            cv_image = cv_image.astype(np.uint8, copy=True)

            # OpenCV (RGB) → PIL Image (RGB)
            image_pil = Image.fromarray(cv_image, mode='RGB')

        except Exception as e:
            self.get_logger().error(f"cv_bridgeまたは画像変換エラー: {e}")
            return

        # =====================================
        # LangSAM 推論を非同期で実行
        # =====================================
        if not self.predicting:
            self.predicting = True
            self.thread_pool.submit(
                self.run_inference,
                image_pil,
                cv_image.copy()
            )

        # =====================================
        # オプティカルフローによるマスク補間
        # =====================================
        with self.lock:
            if self.prev_image is not None and self.prev_masks is not None:
                prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,       # 1フレーム前のグレースケール画像
                    curr_gray,       # 現在のグレースケール画像
                    None,            # 出力初期値（通常None）
                    pyr_scale=0.5,   # ピラミッドスケール：下層画像サイズの縮小率
                    levels=4,        # ピラミッドのレベル数（解像度の段階）
                    winsize=25,      # 各ピクセル周辺の計算ウィンドウサイズ
                    iterations=5,    # 各レベルでの繰り返し回数
                    poly_n=7,        # 多項式展開に使う近傍のサイズ
                    poly_sigma=1.5,  # 多項式展開のガウシアンσ
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN          # オプション（通常は0）
                )

                h, w = flow.shape[:2]
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (map_x + flow[..., 0]).astype(np.float32)
                map_y = (map_y + flow[..., 1]).astype(np.float32)

                warped_masks = np.array([
                    cv2.remap(mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST)
                    for mask in self.prev_masks
                ]).astype(bool)

                if warped_masks.shape[0] > 0 and warped_masks.ndim == 3:
                    annotated_image = draw_image(
                        image_rgb=cv_image,
                        masks=warped_masks,
                        xyxy=self.prev_boxes,
                        probs=self.prev_scores,
                        labels=self.prev_labels
                    )
                else:
                    self.get_logger().warn("補間マスクが空のため、描画をスキップします")
                    annotated_image = cv_image.copy()
                
                # =====================================
                # オプティカルフローのデバック用
                # =====================================
                flow_vis = draw_optical_flow(cv_image, flow)            
                try:
                    flow_msg = self.bridge.cv2_to_imgmsg(flow_vis, encoding='rgb8')
                    self.flow_pub.publish(flow_msg)
                except Exception as e:
                    self.get_logger().error(f"フロー描画画像のパブリッシュ失敗: {e}")

            else:
                annotated_image = cv_image.copy()

        # =====================================
        # 処理済み画像のパブリッシュ
        # =====================================
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            self.mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"マスク画像のパブリッシュ失敗: {e}")

    def run_inference(self, image_pil, image_np):
        try:
            # =====================================
            # LangSAMによるマスク推論の実行
            # =====================================
            start = time.time()
            results = self.model.predict([image_pil], [self.text_prompt])
            elapsed = time.time() - start
            # self.get_logger().info(f"LangSAM 推論完了: {elapsed:.2f}秒")

            # =====================================
            # 推論結果の取得と保存（補間用）
            # =====================================
            masks = np.array(results[0]['masks'])
            boxes = np.array(results[0]['boxes'])
            scores = np.array(results[0]['scores'])
            labels = results[0]['labels']

            with self.lock:
                self.prev_image = image_np.copy()
                self.prev_masks = masks.copy()
                self.prev_boxes = boxes.copy()
                self.prev_scores = scores.copy()
                self.prev_labels = labels.copy()

        except Exception as e:
            self.get_logger().error(f"LangSAM推論エラー: {e}")
        finally:
            self.predicting = False


def draw_optical_flow(image, flow, step=16):
    """
    オプティカルフローベクトルを画像上に矢印で描画する

    Args:
        image: ベースとなるRGB画像 (OpenCV形式, shape=(H, W, 3))
        flow: オプティカルフロー結果 (shape=(H, W, 2), dtype=float32)
        step: 描画する間隔（粗さ）ピクセル単位
    Returns:
        flow_vis: 矢印を描画した画像
    """
    flow_vis = image.copy()

    h, w = flow.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            end_x = int(x + dx)
            end_y = int(y + dy)
            cv2.arrowedLine(
                flow_vis,
                (x, y), (end_x, end_y),
                color=(0, 255, 0),  # 緑
                thickness=1,
                tipLength=0.3       # 矢印の先端の長さ
            )
    return flow_vis


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
