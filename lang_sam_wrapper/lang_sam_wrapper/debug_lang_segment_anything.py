import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2

from PIL import Image
import numpy as np
import time
import os
import torch
import warnings

from lang_sam import LangSAM
from lang_sam.utils import draw_image


class LangSAMNode(Node):
    def __init__(self):
        super().__init__('lang_sam_node')

        # ========================================
        # パラメータの宣言と取得
        # ========================================
        self.sam_model = self.declare_and_get_parameter('sam_model', 'sam2.1_hiera_small')
        self.text_prompt = self.declare_and_get_parameter('text_prompt', 'car. wheel.')

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # ========================================
        # CUDA設定とモデル初期化
        # ========================================
        self._configure_cuda_environment()
        
        try:
            self.model = LangSAM(sam_type=self.sam_model)
            self.get_logger().info("LangSAMモデルの初期化が完了しました")
        except Exception as e:
            self.get_logger().error(f"LangSAMモデルの初期化に失敗しました: {repr(e)}")
            self.model = None
            
        self.bridge = CvBridge()

        # ========================================
        # FPS計測用変数の初期化
        # ========================================
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0

        # ========================================
        # ROS2 通信の設定（購読・配信）
        # ========================================
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10)

        self.mask_pub = self.create_publisher(
            ROSImage, '/image_sam', 10)

        self.get_logger().info("LangSAMNode 起動完了")

    def declare_and_get_parameter(self, name, default_value):
        """パラメータを宣言して取得するユーティリティ関数"""
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).get_parameter_value().string_value

    def image_callback(self, msg):
        self.update_fps()

        try:
            # モデルが初期化されていない場合はスキップ
            if self.model is None:
                self.get_logger().warn("LangSAMモデルが初期化されていません。処理をスキップします。")
                return
                
            # ROS画像メッセージ → OpenCV画像 (RGB)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return
            cv_image = cv_image.astype(np.uint8, copy=True)

            # OpenCV → PIL形式へ変換
            image_pil = Image.fromarray(cv_image, mode='RGB')

            # セグメンテーション推論（エラーハンドリング強化）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                try:
                    results = self.model.predict([image_pil], [self.text_prompt])
                except RuntimeError as cuda_error:
                    if "No available kernel" in str(cuda_error) or "CUDA" in str(cuda_error):
                        self.get_logger().warn(f"CUDA エラーが発生しました。CPUモードで再試行します: {cuda_error}")
                        self._force_cpu_mode()
                        results = self.model.predict([image_pil], [self.text_prompt])
                    else:
                        raise cuda_error

            # 結果描画および送信
            self.publish_annotated_image(cv_image, results)

        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {repr(e)}")

    def update_fps(self):
        """FPSを1秒ごとに更新"""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now

    def publish_annotated_image(self, cv_image, results):
        """推論結果を描画して配信"""
        try:
            if not results or len(results) == 0:
                return
                
            first_result = results[0]
            if 'masks' not in first_result or len(first_result['masks']) == 0:
                return
                
            annotated_image = draw_image(
                image_rgb=cv_image,
                masks=np.array(first_result['masks']),
                xyxy=np.array(first_result['boxes']),
                probs=np.array(first_result.get('probs', first_result.get('scores', [1.0] * len(first_result['masks'])))),
                labels=first_result['labels']
            )

            # FPS情報を画像に描画
            cv2.putText(
                annotated_image,
                f"FPS: {self.fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # OpenCV → ROSメッセージ変換および送信
            mask_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            mask_msg.header.stamp = self.get_clock().now().to_msg()
            mask_msg.header.frame_id = 'camera_frame'
            self.mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"マスク描画・送信エラー: {repr(e)}")
    
    def _configure_cuda_environment(self) -> None:
        """CUDA環境の設定とワーニング抑制"""
        try:
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
            if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
                os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
            warnings.filterwarnings("ignore", category=UserWarning)
            
            if torch.cuda.is_available():
                self.get_logger().info(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
                torch.cuda.empty_cache()
            else:
                self.get_logger().info("CUDAが利用できません。CPUモードで実行します。")
        except Exception as e:
            self.get_logger().warn(f"CUDA設定エラー: {repr(e)}")
    
    def _force_cpu_mode(self) -> None:
        """CPUモードに強制的に切り替える"""
        try:
            if hasattr(self.model, 'sam') and hasattr(self.model.sam, 'to'):
                self.model.sam.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.get_logger().info("CPUモードに切り替えました")
        except Exception as e:
            self.get_logger().error(f"CPUモード切り替えエラー: {repr(e)}")


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
