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
from collections import deque
from dataclasses import dataclass
from typing import Optional
from PIL import Image as PILImage

# LangSAM統合トラッカーコンポーネント
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lang_sam'))
from lang_sam_tracker import LangSAMTracker
from lang_sam.utils import draw_image


@dataclass
class FrameData:
    """タイムスタンプ付きフレームデータ構造"""
    image: np.ndarray
    timestamp: float


class LangSAMTrackerNode(Node):
    """統合パイプライン: GroundingDINO → CSRTトラッキング → SAM2セグメンテーション"""
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        # 基本設定
        self.bridge = CvBridge()
        self.last_gdino_time = 0.0
        self.gdino_processing = False
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        
        # 現在フレーム保存（遅延解消のため）
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # ROS2パラメータ設定
        self._setup_parameters()
        
        # CUDA環境設定
        self._setup_cuda()
        
        # LangSAMトラッカー初期化（安全版）
        try:
            self.get_logger().info("SAM2モデル読み込み開始...")
            self.tracker = LangSAMTracker(sam_type=self.sam_model)
            self.get_logger().info("SAM2モデル読み込み完了")
            
            self.get_logger().info("追跡対象設定開始...")
            self.tracker.set_tracking_targets(self.tracking_targets)
            self.get_logger().info("追跡対象設定完了")
        except Exception as e:
            self.get_logger().error(f"LangSAMトラッカー初期化失敗: {e}")
            import traceback
            self.get_logger().error(f"詳細: {traceback.format_exc()}")
            raise
        
        # ROS2通信設定
        self._setup_ros_communication()
        
        self.get_logger().info(f"LangSAMトラッカー初期化完了: {self.sam_model}")
    
    def _setup_parameters(self):
        """ROS2パラメータ宣言・読み込み"""
        # モデル設定
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.2)
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # 実行設定
        self.declare_parameter('gdino_interval_seconds', 1.0)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('enable_sam', True)
        
        # トピック設定
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # パラメータ値取得
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
    
    def _setup_cuda(self):
        """CUDA環境設定"""
        warnings.filterwarnings("ignore", category=UserWarning)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_ros_communication(self):
        """ROS2通信設定"""
        # 入力サブスクライバー
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )
        
        # 出力パブリッシャー（GroundingDINO、CSRT、SAM2）
        self.gdino_pub = self.create_publisher(Image, self.gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, self.csrt_topic, 10)
        self.sam_pub = self.create_publisher(Image, self.sam_topic, 10)
    
    def destroy_node(self):
        """スレッドプールのクリーンアップ付きシャットダウン"""
        self.thread_pool.shutdown(wait=True)
        super().destroy_node()
    
    def image_callback(self, msg: Image):
        """メイン処理（遅延解消版）"""
        try:
            # 基本的な画像変換
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            
            # 現在フレーム更新（遅延解消のため履歴保存廃止）
            with self.frame_lock:
                self.current_frame = image.copy()
            
            # GroundingDINO実行判定（時間ベース）
            with self.lock:
                should_run_gdino = (
                    current_time - self.last_gdino_time >= self.gdino_interval_seconds 
                    and not self.gdino_processing
                )
            
            # GroundingDINO非同期実行（現在フレーム使用）
            if should_run_gdino:
                with self.lock:
                    self.gdino_processing = True
                    self.last_gdino_time = current_time
                # 現在フレームのコピーを非同期処理に渡す
                self.thread_pool.submit(self._async_gdino_processing, image.copy())
            
            # CSRT+SAM2毎フレーム実行（現在フレーム使用）
            if self.enable_tracking:
                self._run_csrt_and_sam(image)
            else:
                # トラッキング無効の場合は元画像配信
                self._publish_original_image(image)
                
        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {e}")
    
    def _async_gdino_processing(self, frame: np.ndarray):
        """GroundingDINO非同期処理（遅延解消版）"""
        try:
            # トラッカーの安全性チェック
            if not hasattr(self, 'tracker') or self.tracker is None:
                self.get_logger().error("GroundingDINO: トラッカーが初期化されていません")
                return
                
            # フレーム安全性チェック
            if frame is None or frame.size == 0:
                self.get_logger().error("GroundingDINO: 無効なフレーム")
                return
            
            # BGR→RGB変換（PIL用）
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(image_rgb)
            except Exception as e:
                self.get_logger().error(f"GroundingDINO: 画像変換エラー: {e}")
                return
            
            # GroundingDINO実行（安全版）
            try:
                with warnings.catch_warnings(), torch.no_grad():
                    warnings.simplefilter("ignore")
                    results = self.tracker.predict_with_tracking(
                        images_pil=[pil_image],
                        texts_prompt=[self.text_prompt],
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        update_trackers=True,
                        run_sam=False
                    )
            except Exception as e:
                self.get_logger().error(f"GroundingDINO: モデル実行エラー: {e}")
                # エラー時も元画像配信
                msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                self.gdino_pub.publish(msg)
                return
            
            # 検出結果処理
            if results and len(results) > 0:
                result = results[0]
                boxes = result.get('boxes', [])
                labels = result.get('labels', [])
                scores = result.get('scores', [])
                
                # GroundingDINO結果配信（現在フレーム使用）
                self._publish_detection(frame, boxes, labels, scores, self.gdino_pub)
            else:
                # 検出結果なしの場合も元画像配信
                msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                self.gdino_pub.publish(msg)
                
        except Exception as e:
            self.get_logger().error(f"GroundingDINO: 予期しないエラー: {e}")
        finally:
            with self.lock:
                self.gdino_processing = False
    
    # 以下のメソッドは遅延解消のため無効化
    # def _get_frame_at_time(self, target_time: float) -> Optional[np.ndarray]:
    #     """指定時刻最近フレーム取得（無効化済み）"""
    #     pass
    # 
    # def _fast_forward_trackers(self):
    #     """トラッカー早送り処理（無効化済み）"""
    #     pass
    
    def _run_csrt_and_sam(self, image: np.ndarray):
        """CSRT+SAM2毎フレーム処理（安全版）"""
        try:
            # 画像の安全性チェック
            if image is None or image.size == 0:
                self.get_logger().error("無効な画像データ")
                return
                
            # トラッカーの安全性チェック
            if not hasattr(self, 'tracker') or self.tracker is None:
                self.get_logger().error("トラッカーが初期化されていません")
                # 元画像配信
                msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                self.csrt_pub.publish(msg)
                self.sam_pub.publish(msg)
                return
                
            # CSRT+SAM2実行
            if self.enable_sam:
                result = self.tracker.update_trackers_with_sam(image)
            else:
                result = self.tracker.update_trackers_only(image)
            
            if result is None:
                self.get_logger().error("トラッカー結果がNone")
                msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                self.csrt_pub.publish(msg)
                self.sam_pub.publish(msg)
                return
            
            boxes = result.get('boxes', [])
            labels = result.get('labels', [])
            masks = result.get('masks', [])
            mask_scores = result.get('mask_scores', [])
            
            # CSRT結果配信
            self._publish_tracking(image, boxes, labels, self.csrt_pub)
            
            # SAM2結果配信
            if self.enable_sam and len(boxes) > 0:
                self._publish_segmentation(image, masks, mask_scores, boxes, labels, self.sam_pub)
            elif self.enable_sam:
                # SAM2有効だが追跡対象なしの場合は元画像配信
                msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                self.sam_pub.publish(msg)
                
        except Exception as e:
            self.get_logger().error(f"CSRT+SAMエラー: {e}")
            # エラー時も画像配信を継続
            try:
                msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                self.csrt_pub.publish(msg)
                self.sam_pub.publish(msg)
            except Exception:
                pass
    
    def _publish_original_image(self, image: np.ndarray):
        """元画像を全チャンネルに配信（トラッキング無効時）"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            self.gdino_pub.publish(msg)
            self.csrt_pub.publish(msg)
            self.sam_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"元画像配信エラー: {e}")
    
    def _publish_detection(self, image: np.ndarray, boxes, labels, scores, publisher):
        """GroundingDINO検出結果配信"""
        try:
            if len(boxes) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # Supervision描画（BGR→RGB→BGR変換）
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
            
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            publisher.publish(self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8'))
            
        except Exception as e:
            self.get_logger().error(f"検出結果配信エラー: {e}")
    
    def _publish_tracking(self, image: np.ndarray, boxes, labels, publisher):
        """CSRT追跡結果配信"""
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
    
    def _publish_segmentation(self, image: np.ndarray, masks, mask_scores, boxes, labels, publisher):
        """SAM2セグメンテーション結果配信"""
        try:
            # 入力データ安全性チェック
            boxes_array = np.array(boxes) if len(boxes) > 0 else np.array([])
            if boxes_array.size == 0 or len(labels) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # mask_scores安全処理
            if isinstance(mask_scores, (list, tuple)):
                mask_scores = np.array(mask_scores) if len(mask_scores) > 0 else np.ones(len(boxes))
            elif isinstance(mask_scores, np.ndarray) and mask_scores.size == 0:
                mask_scores = np.ones(len(boxes))
            elif not isinstance(mask_scores, np.ndarray):
                mask_scores = np.ones(len(boxes))
            
            # 形状調整
            if mask_scores.ndim == 0:
                mask_scores = np.array([float(mask_scores)])
            elif mask_scores.ndim > 1:
                mask_scores = mask_scores.flatten()
            
            if len(mask_scores) != len(boxes):
                mask_scores = np.ones(len(boxes))
            
            # masks安全処理
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
            # エラー時は元画像配信
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