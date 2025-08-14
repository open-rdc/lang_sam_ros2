#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from PIL import Image as PILImage

# 統合後のモジュール
from lang_sam.tracker_utils import LangSAMTracker, CSRTFrameManager
from lang_sam.utils import draw_image

# ROS2関係のユーティリティ
from lang_sam_wrapper.utils import TrackerParameterManager, ImagePublisher


class LangSAMTrackerNode(Node):
    """リアルタイム物体検出・追跡・セグメンテーション統合ノード
    
    技術アーキテクチャ:
    - GroundingDINO: 自然言語プロンプトを用いたゼロショット物体検出（1Hz間隔でGPU推論）
    - CSRT: 高精度なビジュアルトラッキングアルゴリズム（毎フレーム30Hz実行）
    - SAM2: セマンティックセグメンテーション（トラッキング結果に基づく高精度マスク生成）
    
    システム設計: 検出処理とトラッキング処理を分離し、リアルタイム性能を確保
    """
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        # ROS2メッセージブリッジとスレッド管理の初期化
        # CvBridge: OpenCV画像とROS Image メッセージ間の変換を担当
        self.bridge = CvBridge()
        self.image_publisher = ImagePublisher(self)
        
        # GroundingDINO実行間隔制御：重い推論処理の頻度調整
        self.last_gdino_time = 0.0
        self.gdino_processing = False
        
        # スレッドセーフティ確保：GroundingDINOとCSRTの並行処理制御
        self.lock = threading.Lock()
        # 非同期推論専用スレッドプール：GroundingDINO処理をメインループから分離
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        
        # CSRT復旧機能用フレームバッファマネージャー
        # 目的: トラッキング失敗時の時間遡行による復旧処理実現
        self.csrt_frame_manager = None
        
        # ROS2パラメータサーバーからの設定読み込み
        # 技術的理由: 実行時パラメータ調整とマルチ環境対応のため
        param_manager = TrackerParameterManager(self)
        params = param_manager.initialize_parameters()
        self._load_parameters(params)
        
        # CUDA環境設定
        self._setup_cuda_environment()
        
        # LangSAMトラッカー初期化
        self._initialize_tracker()
        
        # CSRTフレーム機能初期化（パラメータに基づく）
        self._initialize_csrt_features()
        
        # ROS2通信設定
        self._setup_communication()
        
        self.get_logger().info(f"LangSAMトラッカー初期化完了: {self.sam_model}")
    
    def _load_parameters(self, params: dict):
        """パラメータ読み込み"""
        self.sam_model = params['sam_model']
        self.text_prompt = params['text_prompt']
        self.box_threshold = params['box_threshold']
        self.text_threshold = params['text_threshold']
        self.tracking_targets = params['tracking_targets']
        self.gdino_interval_seconds = params['gdino_interval_seconds']
        self.input_topic = params['input_topic']
        self.gdino_topic = params['gdino_topic']
        self.csrt_topic = params['csrt_topic']
        self.sam_topic = params['sam_topic']
        self.bbox_margin = params['bbox_margin']
        self.bbox_min_size = params['bbox_min_size']
        self.tracker_min_size = params['tracker_min_size']
        
        # CSRTフレーム機能パラメータ（統合設定）
        self.enable_csrt_recovery = params['enable_csrt_recovery']
        self.frame_buffer_duration = params['frame_buffer_duration']
        self.time_travel_seconds = params['time_travel_seconds']
        self.fast_forward_frames = params['fast_forward_frames']
        self.recovery_attempt_frames = params['recovery_attempt_frames']
        
        # CSRT内部アルゴリズムパラメータ（24項目の詳細調整）
        # 目的：HOG特徴量、色特徴量、スケール探索などの最適化
        self.csrt_params = {k: v for k, v in params.items() if k.startswith('csrt_')}
    
    def _setup_cuda_environment(self):
        """GPU計算環境の最適化設定
        
        技術的処理：
        - PyTorchのUserWarning抑制（開発環境での冗長ログ削減）
        - CUDAキャッシュクリア（GPUメモリリーク防止）
        - GPU利用可能性確認と初期化
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # GPU VRAM初期化
    
    def _initialize_tracker(self):
        """統合AI推論パイプラインの初期化
        
        技術的構成：
        1. SAM2モデル（hiera_tiny/small/base/large）の選択的読み込み
        2. GroundingDINOによるゼロショット物体検出機能
        3. CSRTトラッカーによる高精度追跡機能
        4. 各コンポーネント間の連携インターフェース構築
        """
        try:
            self.tracker = LangSAMTracker(sam_type=self.sam_model)
            self.tracker.set_tracking_targets(self.tracking_targets)
            self.tracker.set_tracking_config({
                'bbox_margin': self.bbox_margin,
                'bbox_min_size': self.bbox_min_size,
                'tracker_min_size': self.tracker_min_size
            })
            # CSRT内部アルゴリズムの詳細パラメータ調整（24項目）
            # 技術的目的：HOG特徴量、色特徴量、スケール探索の最適化
            if self.csrt_params:
                self.tracker.set_csrt_params(self.csrt_params)
        except Exception as e:
            self.get_logger().error(f"統合AI推論システム初期化失敗: {e}")
            raise
    
    def _initialize_csrt_features(self):
        """トラッキング復旧用フレームバッファシステム初期化
        
        技術的機能：
        - リングバッファによる過去フレーム履歴保持
        - 時間インデックス管理による高速フレーム検索
        - トラッキング失敗検出時の自動復旧トリガー
        """
        if self.enable_csrt_recovery:
            try:
                self.csrt_frame_manager = CSRTFrameManager(buffer_duration=self.frame_buffer_duration)
                self.get_logger().info(
                    f"CSRT復旧機能初期化完了: バッファ{self.frame_buffer_duration}秒, "
                    f"時間復旧{self.time_travel_seconds}秒, 早送り{self.fast_forward_frames}フレーム"
                )
            except Exception as e:
                self.get_logger().error(f"CSRT復旧機能初期化失敗: {e}")
                self.enable_csrt_recovery = False
                self.csrt_frame_manager = None
        else:
            self.get_logger().info("CSRT復旧機能は無効化されています")
    
    def _setup_communication(self):
        """ROS2ノード間通信インターフェースの構築
        
        通信設計：
        - 入力：ZEDカメラからのRGB画像ストリーム（30Hz）
        - 出力：3チャンネル画像配信（検出/追跡/セグメンテーション結果）
        - QoSプロファイル：リアルタイム性優先の設定
        """
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
        """メイン画像処理（フレームバッファリング対応）"""
        try:
            # ROS Image → OpenCV numpy変換（BGR色空間）
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            
            # CSRTフレーム保存（統合復旧機能）
            if self.enable_csrt_recovery and self.csrt_frame_manager:
                self.csrt_frame_manager.process_incoming_frame(image)
            
            # GroundingDINO実行判定（時間ベース・非同期）
            with self.lock:
                should_run_gdino = (
                    current_time - self.last_gdino_time >= self.gdino_interval_seconds 
                    and not self.gdino_processing
                )
            
            # GroundingDINO非同期実行（バッファリング開始）
            if should_run_gdino:
                with self.lock:
                    self.gdino_processing = True
                    self.last_gdino_time = current_time
                
                # GroundingDINO処理開始（最新フレームで直接実行）
                self.thread_pool.submit(self._async_gdino_processing, image.copy())
            
            # CSRT+SAM2同期実行（30Hz、毎フレーム）
            self._run_csrt_and_sam(image)
                
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
                    update_trackers=False,  # フレームバッファでのみ初期化
                    run_sam=False  # SAM2は毎フレーム実行のため無効化
                )
            
            # 検出結果処理・フレームバッファキャッチアップ
            if results and len(results) > 0:
                result = results[0]
                boxes = result.get('boxes', [])
                labels = result.get('labels', [])
                scores = result.get('scores', [])
                
                # CSRTトラッカー初期化（スレッドセーフ）
                if hasattr(self.tracker, 'coordinator') and hasattr(self.tracker.coordinator, 'tracking_manager'):
                    tracking_manager = self.tracker.coordinator.tracking_manager
                    if tracking_manager:
                        # 複製防止のためロックしてトラッカー初期化
                        with self.lock:
                            tracking_manager.initialize_trackers(boxes, labels, frame)
                        self.get_logger().info(f"CSRTトラッカー初期化完了: {len(boxes)}オブジェクト")
                
                # 検出フレームでの結果配信（同期表示）
                self._publish_detection(frame, boxes, labels, scores, self.gdino_pub)
            else:
                # 検出なし時は元フレーム配信
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
            
            # トラッカーの有効性確認（復旧処理の前提条件）
            if not self.tracker.has_active_tracking:
                self._publish_fallback_images(image)
                return
            
            # CSRT+SAM2統合実行（複製防止のためロック）
            with self.lock:
                result = self.tracker.update_trackers_with_sam(image)
            
            # トラッキング失敗時の復旧処理（統合設定対応）
            if (result is None or len(result.get('boxes', [])) == 0) and self.enable_csrt_recovery and self.tracker.has_active_tracking:
                recovery_result = self._attempt_tracking_recovery(image)
                if recovery_result:
                    result = recovery_result
                else:
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
            if len(boxes) > 0:
                self._publish_segmentation(image, masks, mask_scores, boxes, labels, self.sam_pub)
            else:
                self.sam_pub.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                
        except Exception as e:
            self.get_logger().error(f"CSRT+SAMエラー: {e}")
            self._publish_fallback_images(image)
    
    
    def _publish_fallback_images(self, image: np.ndarray):
        """フォールバック画像配信（エラー時CSRT・SAM継続）"""
        self._publish_to_multiple([self.csrt_pub, self.sam_pub], image)
    
    def _publish_to_multiple(self, publishers, image: np.ndarray, error_msg: str = None):
        """複数パブリッシャーに画像配信"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            for publisher in publishers:
                publisher.publish(msg)
        except Exception as e:
            if error_msg:
                self.get_logger().error(f"{error_msg}: {e}")
    
    def _draw_results(self, image: np.ndarray, masks, boxes, labels, scores):
        """結果描画"""
        if len(boxes) == 0:
            return image
            
        # マスク処理（簡素化）
        if len(masks) == 0:
            height, width = image.shape[:2]
            masks = np.zeros((len(boxes), height, width), dtype=bool)
        
        # 一回の変換で描画実行
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgb = draw_image(
            image_rgb=image_rgb,
            masks=np.array(masks),
            xyxy=np.array(boxes),
            probs=np.array(scores),
            labels=labels
        )
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    def _publish_detection(self, image: np.ndarray, boxes, labels, scores, publisher):
        """GroundingDINO検出結果配信（BoundingBox + ラベル描画）"""
        self._publish_with_results(image, publisher, boxes, labels, scores, [], "検出結果配信エラー")
    
    def _publish_tracking(self, image: np.ndarray, boxes, labels, publisher):
        """CSRT追跡結果配信（リアルタイム追跡BOX描画）"""
        dummy_scores = np.ones(len(boxes)) if len(boxes) > 0 else []
        self._publish_with_results(image, publisher, boxes, labels, dummy_scores, [], "追跡結果配信エラー")
    
    def _publish_with_results(self, image: np.ndarray, publisher, boxes, labels, scores, masks, error_msg: str):
        """結果付き画像配信（共通処理）"""
        try:
            if len(boxes) == 0:
                publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
                return
            
            result_bgr = self._draw_results(image, masks, boxes, labels, scores)
            publisher.publish(self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8'))
            
        except Exception as e:
            self.get_logger().error(f"{error_msg}: {e}")
            publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
    
    def _publish_segmentation(self, image: np.ndarray, masks, mask_scores, boxes, labels, publisher):
        """SAM2セグメンテーション結果配信（高精度マスク描画）"""
        validated_data = self._validate_segmentation_data(masks, mask_scores, boxes, labels)
        if not validated_data:
            publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
            return
        
        masks_array, scores_array, boxes_array = validated_data
        self._publish_with_results(image, publisher, boxes_array, labels, scores_array, masks_array, "セグメンテーション配信エラー")
    
    def _validate_segmentation_data(self, masks, mask_scores, boxes, labels):
        """セグメンテーションデータ検証"""
        boxes_array = np.array(boxes) if len(boxes) > 0 else np.array([])
        if boxes_array.size == 0 or len(labels) == 0:
            return None
        
        # mask_scores配列安全処理
        if not isinstance(mask_scores, np.ndarray) or mask_scores.size == 0:
            scores_array = np.ones(len(boxes))
        elif mask_scores.ndim != 1 or len(mask_scores) != len(boxes):
            scores_array = np.ones(len(boxes))
        else:
            scores_array = mask_scores
        
        # masks配列安全処理
        if isinstance(masks, (list, tuple)) and len(masks) > 0:
            masks_array = np.array(masks)
        elif isinstance(masks, np.ndarray) and masks.size > 0:
            masks_array = masks
        else:
            masks_array = np.array([])
        
        return masks_array, scores_array, boxes_array
    
    def _attempt_tracking_recovery(self, current_image: np.ndarray) -> Optional[Dict]:
        """トラッキング失敗時の時間さかのぼり・早送り復旧処理（統合設定対応）"""
        try:
            # フレームマネージャーとトラッカーの有効性確認
            if (not self.csrt_frame_manager or 
                not hasattr(self.tracker, 'coordinator') or 
                not hasattr(self.tracker.coordinator, 'tracking_manager') or
                not self.tracker.coordinator.tracking_manager):
                return None
                
            tracking_manager = self.tracker.coordinator.tracking_manager
            
            # アクティブトラッカーが存在しない場合は復旧不可
            if not tracking_manager.has_active_trackers():
                return None
                
            self.get_logger().debug("トラッキング失敗 - 時間さかのぼり復旧開始")
            
            # 設定時間で過去フレーム取得
            past_frame_data = self.csrt_frame_manager.get_past_frame(seconds_ago=self.time_travel_seconds)
            if past_frame_data:
                past_frame, timestamp = past_frame_data
                self.get_logger().debug(f"{self.time_travel_seconds}秒前フレームで復旧試行: {timestamp}")
                
                # 過去フレームでトラッキング実行
                past_result = tracking_manager.update_all_trackers(past_frame)
                
                if past_result and len(past_result) > 0:
                    self.get_logger().debug("過去フレームで復旧成功 - 早送りキャッチアップ開始")
                    
                    # 早送り機能で現在フレームまでキャッチアップ
                    return self._fast_forward_catchup(tracking_manager, current_image)
            
            # 復旧失敗
            self.get_logger().debug("時間さかのぼり復旧失敗")
            return None
            
        except Exception as e:
            self.get_logger().error(f"復旧処理エラー: {e}")
            return None
    
    def _fast_forward_catchup(self, tracking_manager, target_image: np.ndarray) -> Optional[Dict]:
        """早送り機能で現在フレームまでキャッチアップ（統合設定対応）"""
        try:
            # 設定フレーム数で高速キャッチアップ
            recent_frames = self.csrt_frame_manager.get_recent_frames(count=self.fast_forward_frames)
            
            if len(recent_frames) > 1:
                self.get_logger().info(f"早送りキャッチアップ: {len(recent_frames)}フレーム処理")
                
                # 高速バッチ処理（最新N フレーム使用）
                process_frames = recent_frames[-self.recovery_attempt_frames:]
                for frame, timestamp in process_frames:
                    tracking_manager.update_all_trackers(frame)
                
                # 最終的に現在フレームで結果取得
                final_result = tracking_manager.update_all_trackers(target_image)
                
                if final_result and len(final_result) > 0:
                    self.get_logger().info("早送りキャッチアップ成功")
                    
                    # SAM2付きで最終結果生成
                    return self.tracker.update_trackers_with_sam(target_image)
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"早送りキャッチアップエラー: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = LangSAMTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()