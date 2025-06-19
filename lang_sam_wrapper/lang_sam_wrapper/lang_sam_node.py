"""
LangSAM ROS2 ノード

テキストプロンプトを使用してセグメンテーションを実行するROSノード。
言語による指示で物体検出とセグメンテーションを行い、結果を可視化して配信する。
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2

from PIL import Image
import numpy as np

from lang_sam import LangSAM
from lang_sam.utils import draw_image
from lang_sam_msgs.msg import SamMasks


class LangSAMNode(Node):
    """
    LangSAMノード：テキストプロンプトによるセグメンテーション処理
    
    テキストによる指示で物体を検出・セグメンテーションし、
    結果を可視化して配信するROSノード。
    """
    
    def __init__(self):
        super().__init__('lang_sam_node')

        # パラメータの宣言と取得
        self.sam_model = self.declare_and_get_parameter('sam_model', 'sam2.1_hiera_small')
        self.text_prompt = self.declare_and_get_parameter('text_prompt', 'car. wheel.')

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # モデルおよびユーティリティの初期化
        self.model = LangSAM(sam_type=self.sam_model)
        self.bridge = CvBridge()


        # ROS2 通信の設定（購読・配信）
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10)

        self.mask_pub = self.create_publisher(
            ROSImage, '/image_sam', 10)
        
        self.sam_masks_pub = self.create_publisher(
            SamMasks, '/sam_masks', 10)

        self.get_logger().info("LangSAMNode 起動完了")

    def declare_and_get_parameter(self, name, default_value):
        """パラメータを宣言して取得するユーティリティ関数"""
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).get_parameter_value().string_value

    def image_callback(self, msg):
        """画像コールバック関数：受信した画像に対してセグメンテーション処理を実行"""
        try:
            # ROS画像メッセージ → OpenCV画像 (RGB)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return
            cv_image = cv_image.astype(np.uint8, copy=True)

            # OpenCV → PIL形式へ変換
            image_pil = Image.fromarray(cv_image, mode='RGB')

            # セグメンテーション推論
            results = self.model.predict([image_pil], [self.text_prompt])

            # 結果描画および送信
            self.publish_annotated_image(cv_image, results)
            self.publish_sam_masks(results)

        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {repr(e)}")


    def publish_annotated_image(self, cv_image, results):
        """推論結果を描画して配信する関数"""
        try:
            masks = np.array(results[0]['masks'])
            boxes = np.array(results[0]['boxes'])
            probs = np.array(results[0].get('probs', [1.0] * len(results[0]['masks'])))
            labels = results[0]['labels']
            
            annotated_image = draw_image(
                image_rgb=cv_image,
                masks=masks,
                xyxy=boxes,
                probs=probs,
                labels=labels
            )

            # OpenCV → ROSメッセージ変換および送信
            mask_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            self.mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"マスク描画・送信エラー: {repr(e)}")
    
    def publish_sam_masks(self, results):
        """マスク情報をSamMasksメッセージで配信する関数"""
        try:
            sam_masks_msg = SamMasks()
            sam_masks_msg.header.stamp = self.get_clock().now().to_msg()
            sam_masks_msg.header.frame_id = 'camera_frame'
            
            masks = np.array(results[0]['masks'])
            boxes = np.array(results[0]['boxes'])
            probs = np.array(results[0].get('probs', [1.0] * len(results[0]['masks'])))
            labels = results[0]['labels']
            
            sam_masks_msg.labels = labels
            sam_masks_msg.boxes = boxes.flatten().tolist()  # flatten to 1D array
            sam_masks_msg.probs = probs.tolist()
            
            for mask in masks:
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
                sam_masks_msg.masks.append(mask_msg)
            
            self.sam_masks_pub.publish(sam_masks_msg)
            
        except Exception as e:
            self.get_logger().error(f"SamMasks送信エラー: {repr(e)}")