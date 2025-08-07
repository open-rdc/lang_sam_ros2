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
from typing import Optional, List
from PIL import Image as PILImage

# LangSAM integrated tracker components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lang_sam'))
from lang_sam_tracker import LangSAMTracker
from lang_sam.utils import draw_image


@dataclass
class FrameData:
    """フレームデータ構造（タイムスタンプ付き）"""
    image: np.ndarray
    timestamp: float


class LangSAMTrackerNode(Node):
    """Integrated pipeline: GroundingDINO → CSRT tracking → SAM2 segmentation"""
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        # Core components initialization
        self.bridge = CvBridge()
        self.last_gdino_time = 0.0
        self.gdino_processing = False
        self.gdino_start_time = None  # GDINO処理開始時刻
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        
        # フレームバッファ（3秒履歴、約90フレーム@30fps）
        self.frame_buffer: deque[FrameData] = deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        
        # Load all parameters from config.yaml
        self._declare_all_parameters()
        self._load_all_parameters()
        
        # CUDA environment configuration
        self._setup_cuda_environment()
        
        # Initialize integrated LangSAM tracker
        self.tracker = LangSAMTracker(sam_type=self.sam_model)
        self.tracker.set_tracking_targets(self.tracking_targets)
        
        # ROS2 communication setup (parameterized topics)
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )
        
        # Triple output publishers (parameterized)
        self.gdino_pub = self.create_publisher(Image, self.gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, self.csrt_topic, 10)
        self.sam_pub = self.create_publisher(Image, self.sam_topic, 10)
        
        self.get_logger().info(f"LangSAM Tracker initialized: {self.sam_model}")
    
    def destroy_node(self):
        """Clean shutdown with thread pool cleanup"""
        self.thread_pool.shutdown(wait=True)
        super().destroy_node()
    
    def _declare_all_parameters(self):
        """Declare all ROS2 parameters"""
        # Model configuration
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.2)
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # Execution configuration
        self.declare_parameter('gdino_interval_seconds', 3.0)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('enable_sam', True)
        
        # Topic configuration
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
    
    def _load_all_parameters(self):
        """Load all ROS2 parameters"""
        # Model configuration
        self.sam_model = self.get_parameter('sam_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        self.tracking_targets = self.get_parameter('tracking_targets').value
        
        # Execution configuration
        self.gdino_interval_seconds = self.get_parameter('gdino_interval_seconds').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.enable_sam = self.get_parameter('enable_sam').value
        
        # Topic configuration
        self.input_topic = self.get_parameter('input_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.csrt_topic = self.get_parameter('csrt_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
    
    def _setup_cuda_environment(self):
        """Configure CUDA environment"""
        warnings.filterwarnings("ignore", category=UserWarning)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def image_callback(self, msg: Image):
        """フレームバッファ機能付きメイン処理"""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            
            # フレームバッファに保存（常時）
            with self.buffer_lock:
                self.frame_buffer.append(FrameData(image.copy(), current_time))
            
            # GDINO実行判定
            with self.lock:
                should_run_gdino = (
                    current_time - self.last_gdino_time >= self.gdino_interval_seconds 
                    and not self.gdino_processing
                )
            
            # GDINO非同期処理開始
            if should_run_gdino:
                with self.lock:
                    self.gdino_processing = True
                    self.gdino_start_time = current_time
                    self.last_gdino_time = current_time
                self.thread_pool.submit(self._async_gdino_processing)
            
            # CSRT+SAM2は毎フレーム実行
            if self.enable_tracking:
                self._run_csrt_and_sam(image)
                
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
    
    def _async_gdino_processing(self):
        """GDINO処理 + フレーム履歴遡り + トラッカー早送り"""
        try:
            print("\n=== GDINO処理開始 ===")
            # 処理開始時点のフレーム取得
            start_frame = self._get_frame_at_time(self.gdino_start_time)
            if start_frame is None:
                print("ERROR: 開始時点のフレームが見つからない")
                return
            
            print(f"処理開始時刻: {self.gdino_start_time}")
            print(f"フレームサイズ: {start_frame.shape}")
            
            # GDINO実行（処理開始時点のフレーム使用）
            image_rgb = cv2.cvtColor(start_frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            print(f"PIL画像作成: {pil_image.size}")
            
            print(f"GDINO実行開始 - prompt: '{self.text_prompt}'")
            print(f"閾値: box={self.box_threshold}, text={self.text_threshold}")
            
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
            
            print(f"GDINO実行完了 - 結果数: {len(results) if results else 0}")
            
            # GDINO結果処理
            if results and len(results) > 0:
                result = results[0]
                boxes = result.get('boxes', [])
                labels = result.get('labels', [])
                scores = result.get('scores', [])
                
                print(f"検出結果: {len(boxes)}個のオブジェクト")
                print(f"ラベル: {labels}")
                print(f"スコア: {scores}")
                print(f"bbox: {boxes}")
                
                self._publish_detection(start_frame, boxes, labels, scores, self.gdino_pub)
                
                # トラッカー早送り（処理開始時点→現在まで）
                self._fast_forward_trackers()
            else:
                print("GDINO検出結果なし")
            
            print("=== GDINO処理完了 ===\n")
                
        except Exception as e:
            print(f"GDINO ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
                self.gdino_processing = False
    
    def _get_frame_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """指定時刻に最も近いフレームを取得"""
        with self.buffer_lock:
            if not self.frame_buffer:
                print("ERROR: フレームバッファが空")
                return None
            
            print(f"フレームバッファサイズ: {len(self.frame_buffer)}")
            # 時刻差最小のフレーム検索
            best_frame = min(self.frame_buffer, key=lambda f: abs(f.timestamp - target_time))
            time_diff = abs(best_frame.timestamp - target_time)
            print(f"最適フレーム選択: 時差{time_diff:.3f}秒")
            return best_frame.image
    
    def _fast_forward_trackers(self):
        """トラッカーを処理開始時点から現在まで早送り"""
        if not self.gdino_start_time:
            print("WARNING: GDINO開始時刻が未設定")
            return
        
        with self.buffer_lock:
            # 処理開始時点以降のフレーム抽出
            recent_frames = [f for f in self.frame_buffer if f.timestamp > self.gdino_start_time]
        
        print(f"早送り処理: {len(recent_frames)}フレームを処理")
        # 早送り実行（効率化のため間引き処理）
        processed_count = 0
        for i, frame_data in enumerate(recent_frames):
            if i % 2 == 0:  # 2フレームおきに処理
                self.tracker._update_existing_trackers(frame_data.image)
                processed_count += 1
        print(f"早送り完了: {processed_count}フレーム処理済み")
    
    def _run_csrt_and_sam(self, image: np.ndarray):
        """CSRT+SAM2毎フレーム実行（最適化版）"""
        try:
            # トラッカー状況確認
            tracker_count = len(self.tracker.trackers) if hasattr(self.tracker, 'trackers') else 0
            
            if self.enable_sam:
                result = self.tracker.update_trackers_with_sam(image)
            else:
                result = self.tracker.update_trackers_only(image)
            
            boxes = result.get('boxes', [])
            labels = result.get('labels', [])
            masks = result.get('masks', [])
            mask_scores = result.get('mask_scores', [])
            
            # 定期的にトラッキング状況をログ出力（100フレームに1回）
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 1
            
            if self.frame_count % 100 == 0:
                print(f"フレーム{self.frame_count}: トラッカー{tracker_count}個, 追跡{len(boxes)}個")
            
            # CSRT結果配信
            self._publish_tracking(image, boxes, labels, self.csrt_pub)
            
            # SAM2結果配信
            if self.enable_sam:
                if len(boxes) > 0:
                    self._publish_segmentation(image, masks, mask_scores, boxes, labels, self.sam_pub)
                else:
                    msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                    self.sam_pub.publish(msg)
                
        except Exception as e:
            print(f"CSRT+SAM ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _publish_detection(self, image: np.ndarray, boxes, labels, scores, publisher):
        """GDINO検出結果配信"""
        try:
            if len(boxes) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # draw_imageによる可視化
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
            self.get_logger().error(f"Detection error: {e}")
    
    def _publish_tracking(self, image: np.ndarray, boxes, labels, publisher):
        """CSRT追跡結果配信"""
        try:
            if len(boxes) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # draw_imageによる可視化
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
            self.get_logger().error(f"Tracking error: {e}")
    
    def _publish_segmentation(self, image: np.ndarray, masks, mask_scores, boxes, labels, publisher):
        """SAM2セグメンテーション結果配信（安全な配列処理）"""
        try:
            # 入力データの安全性チェック
            boxes_array = np.array(boxes) if len(boxes) > 0 else np.array([])
            if boxes_array.size == 0 or len(labels) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            # mask_scoresの安全な処理
            try:
                if isinstance(mask_scores, (list, tuple)):
                    if len(mask_scores) == 0:
                        mask_scores = np.ones(len(boxes))
                    else:
                        mask_scores = np.array(mask_scores)
                elif isinstance(mask_scores, np.ndarray):
                    if mask_scores.size == 0:
                        mask_scores = np.ones(len(boxes))
                else:
                    mask_scores = np.ones(len(boxes))
                
                # 形状確認と修正
                if mask_scores.ndim == 0:
                    mask_scores = np.array([float(mask_scores)])
                elif mask_scores.ndim > 1:
                    mask_scores = mask_scores.flatten()
                
                # 長さ調整
                if len(mask_scores) != len(boxes):
                    mask_scores = np.ones(len(boxes))
                    
            except Exception:
                mask_scores = np.ones(len(boxes))
            
            # masksの安全な処理
            try:
                if isinstance(masks, (list, tuple)) and len(masks) > 0:
                    masks_array = np.array(masks)
                    if masks_array.size == 0:
                        masks_array = np.array([])
                elif isinstance(masks, np.ndarray) and masks.size > 0:
                    masks_array = masks
                else:
                    masks_array = np.array([])
            except Exception:
                masks_array = np.array([])
            
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
            self.get_logger().error(f"Segmentation error: {e}")
            # エラー時は元画像を配信
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