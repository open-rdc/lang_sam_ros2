#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import os
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as PILImage

# LangSAM統合トラッカー（GroundingDINO + CSRT + SAM2）
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lang_sam'))
from lang_sam_tracker import LangSAMTracker
from lang_sam.utils import draw_image


class LangSAMTrackerNode(Node):
    """ゼロショット物体検出・追跡・セグメンテーション統合ノード
    
    処理フロー:
    1. GroundingDINO: テキストプロンプトによるゼロショット物体検出（1Hz非同期）
    2. CSRT: Channel and Spatial Reliability Tracking（30Hz同期）
    3. SAM2: Segment Anything Model 2によるセグメンテーション（30Hz同期）
    """
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        # ROS2基盤設定
        self.bridge = CvBridge()  # ROS Image ↔ OpenCV numpy変換
        self.last_gdino_time = 0.0
        self.gdino_processing = False
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)  # GroundingDINO非同期処理用
        
        # ROS2パラメータ設定
        self._setup_parameters()
        
        # CUDA環境設定（GPU利用最適化）
        self._setup_cuda_environment()
        
        # LangSAMトラッカー初期化（SAM2 + GroundingDINO + CSRT統合）
        self._initialize_tracker()
        
        # ROS2通信設定（サブスクライバー・パブリッシャー）
        self._setup_communication()
        
        self.get_logger().info(f"LangSAMトラッカー初期化完了: {self.sam_model}")
    
    def _setup_parameters(self):
        """ROS2パラメータ宣言・読み込み"""
        # AIモデル設定
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.2)
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # 実行周波数設定
        self.declare_parameter('gdino_interval_seconds', 1.0)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('enable_sam', True)
        
        # ROS2トピック設定
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # パラメータ値読み込み
        self.sam_model = self.get_parameter('sam_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        self.tracking_targets = self.get_parameter('tracking_targets').value
        
        self.gdino_interval_seconds = self.get_parameter('gdino_interval_seconds').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.enable_sam = self.get_parameter('enable_sam').value
        
        self.input_topic = self.get_parameter('input_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.csrt_topic = self.get_parameter('csrt_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
    
    def _setup_cuda_environment(self):
        """CUDA環境設定（GPU最適化）"""
        warnings.filterwarnings("ignore", category=UserWarning)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # GPU VRAM初期化
    
    def _initialize_tracker(self):
        """LangSAMトラッカー初期化（SAM2 + GroundingDINO + CSRT）"""
        try:
            self.tracker = LangSAMTracker(sam_type=self.sam_model)
            self.tracker.set_tracking_targets(self.tracking_targets)
        except Exception as e:
            self.get_logger().error(f"トラッカー初期化失敗: {e}")
            raise
    
    def _setup_communication(self):
        """ROS2通信設定（サブスクライバー・パブリッシャー）"""
        # ZED画像入力サブスクライバー
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )
        
        # 3チャンネル結果配信パブリッシャー
        self.gdino_pub = self.create_publisher(Image, self.gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, self.csrt_topic, 10)
        self.sam_pub = self.create_publisher(Image, self.sam_topic, 10)
    
    def destroy_node(self):
        """ThreadPoolExecutorクリーンアップ付きノード終了"""
        self.thread_pool.shutdown(wait=True)
        super().destroy_node()
    
    def image_callback(self, msg: Image):
        """メイン画像処理（リアルタイム追跡・遅延最小化版）"""
        try:
            # ROS Image → OpenCV numpy変換（BGR色空間）
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            
            # GroundingDINO実行判定（時間ベース・非同期）
            with self.lock:
                should_run_gdino = (
                    current_time - self.last_gdino_time >= self.gdino_interval_seconds 
                    and not self.gdino_processing
                )
            
            # GroundingDINO非同期実行（1Hz、遅延最小化のため現在フレーム使用）
            if should_run_gdino:
                with self.lock:
                    self.gdino_processing = True
                    self.last_gdino_time = current_time
                self.thread_pool.submit(self._async_gdino_processing, image.copy())
            
            # CSRT+SAM2同期実行（30Hz、毎フレーム）
            if self.enable_tracking:
                self._run_csrt_and_sam(image)
            else:
                self._publish_original_images(image)
                
        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {e}")
    
    def _async_gdino_processing(self, frame: np.ndarray):
        """GroundingDINO非同期処理（テキストプロンプト→ゼロショット物体検出）"""
        try:
            # フレーム安全性チェック
            if frame is None or frame.size == 0:
                return
            
            # BGR→RGB変換（PILImage用）
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # GroundingDINO実行（CUDA GPU推論）
            with warnings.catch_warnings(), torch.no_grad():
                warnings.simplefilter("ignore")
                results = self.tracker.predict_with_tracking(
                    images_pil=[pil_image],
                    texts_prompt=[self.text_prompt],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    update_trackers=True,  # CSRTトラッカー同時初期化
                    run_sam=False  # SAM2は毎フレーム実行のため無効化
                )
            
            # 検出結果処理・配信
            if results and len(results) > 0:
                result = results[0]
                boxes = result.get('boxes', [])
                labels = result.get('labels', [])
                scores = result.get('scores', [])
                
                self._publish_detection(frame, boxes, labels, scores, self.gdino_pub)
            else:
                # 検出なし時は元画像配信
                self.gdino_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
                
        except Exception as e:
            self.get_logger().error(f"GroundingDINOエラー: {e}")
            # エラー時も元画像配信継続
            self.gdino_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
        finally:
            with self.lock:
                self.gdino_processing = False
    
    def _run_csrt_and_sam(self, image: np.ndarray):
        """CSRT追跡+SAM2セグメンテーション同期実行（30Hz毎フレーム処理）"""
        try:
            # 画像・トラッカー安全性チェック
            if image is None or image.size == 0 or not hasattr(self, 'tracker'):
                self._publish_fallback_images(image)
                return
            
            # CSRT+SAM2統合実行
            if self.enable_sam:
                result = self.tracker.update_trackers_with_sam(image)
            else:
                result = self.tracker.update_trackers_only(image)
            
            if result is None:
                self._publish_fallback_images(image)
                return
            
            # 追跡結果抽出
            boxes = result.get('boxes', [])
            labels = result.get('labels', [])
            masks = result.get('masks', [])
            mask_scores = result.get('mask_scores', [])
            
            # CSRT結果配信（BoundingBox描画）
            self._publish_tracking(image, boxes, labels, self.csrt_pub)
            
            # SAM2結果配信（セグメンテーションマスク描画）
            if self.enable_sam and len(boxes) > 0:
                self._publish_segmentation(image, masks, mask_scores, boxes, labels, self.sam_pub)
            else:
                self.sam_pub.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                
        except Exception as e:
            self.get_logger().error(f"CSRT+SAMエラー: {e}")
            self._publish_fallback_images(image)
    
    def _publish_original_images(self, image: np.ndarray):
        """元画像を全チャンネル配信（トラッキング無効時）"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            self.gdino_pub.publish(msg)
            self.csrt_pub.publish(msg)
            self.sam_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"元画像配信エラー: {e}")
    
    def _publish_fallback_images(self, image: np.ndarray):
        """フォールバック画像配信（エラー時CSRT・SAM継続）"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            self.csrt_pub.publish(msg)
            self.sam_pub.publish(msg)
        except Exception:
            pass
    
    def _publish_detection(self, image: np.ndarray, boxes, labels, scores, publisher):
        """GroundingDINO検出結果配信（BoundingBox + ラベル描画）"""
        try:
            if len(boxes) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # Supervision描画ライブラリ使用（BGR→RGB→BGR変換）
            height, width = image.shape[:2]
            dummy_masks = np.zeros((len(boxes), height, width), dtype=bool)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result_image = draw_image(
                image_rgb=image_rgb,
                masks=dummy_masks,
                xyxy=np.array(boxes),
                probs=np.array(scores),
                labels=labels
            )
            
            # RGB→BGR変換してROS配信
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            publisher.publish(self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8'))
            
        except Exception as e:
            self.get_logger().error(f"検出結果配信エラー: {e}")
            publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
    
    def _publish_tracking(self, image: np.ndarray, boxes, labels, publisher):
        """CSRT追跡結果配信（リアルタイム追跡BOX描画）"""
        try:
            if len(boxes) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # Supervision描画（BGR→RGB→BGR変換）
            height, width = image.shape[:2]
            dummy_masks = np.zeros((len(boxes), height, width), dtype=bool)
            dummy_scores = np.ones(len(boxes))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result_image = draw_image(
                image_rgb=image_rgb,
                masks=dummy_masks,
                xyxy=np.array(boxes),
                probs=dummy_scores,
                labels=labels
            )
            
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            publisher.publish(self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8'))
            
        except Exception as e:
            self.get_logger().error(f"追跡結果配信エラー: {e}")
            publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
    
    def _publish_segmentation(self, image: np.ndarray, masks, mask_scores, boxes, labels, publisher):
        """SAM2セグメンテーション結果配信（高精度マスク描画）"""
        try:
            # 入力データ安全性チェック
            boxes_array = np.array(boxes) if len(boxes) > 0 else np.array([])
            if boxes_array.size == 0 or len(labels) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # mask_scores配列安全処理（SAM2信頼度）
            if not isinstance(mask_scores, np.ndarray) or mask_scores.size == 0:
                mask_scores = np.ones(len(boxes))
            elif mask_scores.ndim != 1 or len(mask_scores) != len(boxes):
                mask_scores = np.ones(len(boxes))
            
            # masks配列安全処理（SAM2セグメンテーションマスク）
            if isinstance(masks, (list, tuple)) and len(masks) > 0:
                masks_array = np.array(masks)
            elif isinstance(masks, np.ndarray) and masks.size > 0:
                masks_array = masks
            else:
                masks_array = np.array([])
            
            # Supervision描画（BGR→RGB→BGR変換）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result_image = draw_image(
                image_rgb=image_rgb,
                masks=masks_array,
                xyxy=boxes_array,
                probs=mask_scores,
                labels=labels
            )
            
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            publisher.publish(self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8'))
            
        except Exception as e:
            self.get_logger().error(f"セグメンテーション配信エラー: {e}")
            publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = LangSAMTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()