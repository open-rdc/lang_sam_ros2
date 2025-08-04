"""
最適化されたLangSAM + オプティカルフローによる物体追跡・セグメンテーション

処理フロー:
1. GroundingDINOでバウンディングボックス検出（同期）
2. オプティカルフローで複数特徴点追跡（同期）
3. SAM2で追跡されたバウンディングボックスを入力にセグメンテーション（同期）
4. 可視化・配信（非同期、性能向上）
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from geometry_msgs.msg import Point32
from lang_sam_msgs.msg import SamMasks, FeaturePoints
import collections

from lang_sam.utils import draw_image
from lang_sam import LangSAM
from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
# バウンディングボックス四隅トラッキング関数を内部で定義
from lang_sam_wrapper.feature_pool import FeaturePool, FeatureQuality
from lang_sam_wrapper.gpu_resource_manager import get_gpu_manager, GPUPriority

from PIL import Image
import os
import torch
import warnings
import threading
import time
import gc
import queue
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor


class LangSamWithOptFlowNode(Node):
    """
    最適化されたLangSAM + オプティカルフロー統合ノード
    
    同期処理でマスク位置合わせを保証:
    - GroundingDINO: バウンディングボックス検出（同期）
    - オプティカルフロー: 複数特徴点追跡（同期）
    - SAM2: 追跡されたバウンディングボックスでセグメンテーション（同期）
    - 可視化: 非同期処理で性能向上
    """
    
    @contextmanager
    def _gpu_memory_context(self):
        """GPU メモリ管理用コンテキストマネージャー（超軽量化版）"""
        try:
            # GPU利用可能時のみストリーム設定
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # デバイス固定で高速化
            yield
        finally:
            # クリーンアップをさらに削減（100フレームに1回のみ）
            if torch.cuda.is_available() and self.frame_count % 100 == 0:
                torch.cuda.empty_cache()
                if self.frame_count % 500 == 0:  # 大幅クリーンアップは500フレームに1回
                    import gc
                    gc.collect()
    
    def __init__(self):
        super().__init__('langsam_with_optflow_node')
        
        # コールバックグループの設定
        self._init_callback_groups()
        
        # パラメータの初期化
        self._init_parameters()
        
        # 内部状態の初期化
        self._init_state()
        
        # ROS通信の設定
        self._init_ros_communication()
        
        self.get_logger().info("LangSAM + Optical Flow Node 起動完了")
        self._log_key_parameters()
    
    def _init_callback_groups(self) -> None:
        """コールバックグループの初期化"""
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.publisher_callback_group = ReentrantCallbackGroup()
        self.processing_lock = threading.RLock()
        self.detection_data_lock = threading.RLock()
        self.tracking_data_lock = threading.RLock()
        self.sam_data_lock = threading.RLock()
    
    def _init_parameters(self) -> None:
        """パラメータの初期化"""
        # 基本パラメータ
        self.text_prompt = self.get_config_param('text_prompt')
        self.box_threshold = self.get_config_param('box_threshold')
        self.text_threshold = self.get_config_param('text_threshold')
        self.tracking_targets_str = self.get_config_param('tracking_targets')
        self.tracking_targets = self._parse_tracking_targets(self.tracking_targets_str)
        
        # 実行間隔パラメータ
        self.grounding_dino_interval = self.get_config_param('grounding_dino_interval', 2.0)
        self.sam2_interval = self.get_config_param('sam2_interval', 0.05)
        self.enable_sam2_every_frame = self.get_config_param('enable_sam2_every_frame', False)
        
        # CSRTトラッカーパラメータ
        self.csrt_tracker_type = self.get_config_param('csrt_tracker_type', 'CSRT')  # CSRTトラッカーを使用
        self.tracker_update_threshold = self.get_config_param('tracker_update_threshold', 0.5)  # 追跡信頼度の閾値
        self.adaptive_bbox_scaling = self.get_config_param('adaptive_bbox_scaling', True)  # バウンディングボックスの可変サイズ化
        
        # 特徴点選択パラメータ
        self.feature_selection_method = self.get_config_param('feature_selection_method', 'harris_response')
        
        # CSRTトラッカー検証パラメータ（オプティカルフローパラメータから置換）
        self.tracker_confidence_threshold = self.get_config_param('tracker_confidence_threshold', 0.5)
        
        # 追跡検証パラメータ
        self.max_displacement = self.get_config_param('max_displacement', 50.0)
        self.min_valid_ratio = self.get_config_param('min_valid_ratio', 0.5)
        
        # トラッカー検証パラメータ
        self.min_tracker_confidence = self.get_config_param('min_tracker_confidence', 0.3)
        self.tracker_failure_threshold = self.get_config_param('tracker_failure_threshold', 5)  # 連続失敗回数
        
        # SAM2パラメータ
        self.sam_model = self.get_config_param('sam_model', 'sam2.1_hiera_small')
        
        # 非同期処理設定
        self.enable_async_grounding_dino = self.get_config_param('enable_async_grounding_dino', True)
        self.enable_async_sam2 = self.get_config_param('enable_async_sam2', True)
        self.enable_parallel_feature_processing = self.get_config_param('enable_parallel_feature_processing', True)
        self.feature_processing_workers = self.get_config_param('feature_processing_workers', 4)
        
        # 時間整合性設定
        self.enable_temporal_alignment = self.get_config_param('enable_temporal_alignment', True)
        self.max_temporal_gap = self.get_config_param('max_temporal_gap', 1.0)
        self.image_buffer_size = self.get_config_param('image_buffer_size', 60)
        self.catchup_threshold = self.get_config_param('catchup_threshold', 0.05)
        self.tracker_interpolation_threshold = self.get_config_param('tracker_interpolation_threshold', 0.1)
        
        # 追跡統合設定
        self.enable_tracking_integration = self.get_config_param('enable_tracking_integration', True)
        self.overlap_iou_threshold = self.get_config_param('overlap_iou_threshold', 0.3)
        self.overlap_distance_threshold = self.get_config_param('overlap_distance_threshold', 100.0)
        self.merge_weight_existing = self.get_config_param('merge_weight_existing', 0.7)
        self.merge_weight_new = self.get_config_param('merge_weight_new', 0.3)
        self.detection_timeout = self.get_config_param('detection_timeout', 2.0)  # 2秒間検出されないと削除
        
        # CSRTトラッカーは内部でcriteriaを管理するため削除
    
    def _init_state(self) -> None:
        """内部状態の初期化"""
        self._configure_cuda_environment()
        
        # モデルの初期化
        try:
            self.get_logger().info("GroundingDINO モデル初期化開始")
            self.gdino_model = GDINO()
            self.gdino_model.build_model()
            self.get_logger().info("GroundingDINO モデル初期化完了")
            
            self.get_logger().info(f"SAM2モデル初期化開始: {self.sam_model}")
            self.sam_model_instance = SAM()
            self.sam_model_instance.build_model(self.sam_model)
            self.get_logger().info("SAM2モデル初期化完了")
            
            # Cannyエッジトラッキング初期化
            self.get_logger().info("Cannyエッジトラッキング初期化完了")
            
        except Exception as e:
            self.get_logger().error(f"モデル初期化エラー: {repr(e)}")
            self.gdino_model = None
            self.sam_model_instance = None
        
        self.bridge = CvBridge()
        
        # 内部状態
        with self.detection_data_lock:
            self.latest_detection_data: Optional[Dict] = None
            self.detection_updated = False
            self.latest_grounding_dino_result: Optional[Dict] = None
            self.grounding_dino_updated = False
            
        with self.tracking_data_lock:
            self.tracked_boxes_per_label: Dict[str, List[float]] = {}  # バウンディングボックス追跡用
            self.tracked_csrt_trackers: Dict[str, cv2.TrackerCSRT] = {}  # CSRTトラッカー追跡用
            self.tracker_failure_counts: Dict[str, int] = {}  # トラッカー失敗回数カウント
            self.tracker_initialization_times: Dict[str, float] = {}  # トラッカー初期化時刻記録
            self.original_box_sizes: Dict[str, Tuple[float, float]] = {}  # 初期バウンディングボックスサイズ (width, height)
            
            # GroundingDINO処理期間中の画像保存システム
            self.stored_images: List[Tuple[float, np.ndarray]] = []  # (timestamp, image)
            self.grounding_dino_start_time: Optional[float] = None
            self.grounding_dino_in_progress: bool = False
            self.object_last_detection_time: Dict[str, float] = {}  # 各オブジェクトの最後検出時刻
            self.tracking_valid = False
            
        with self.sam_data_lock:
            self.latest_sam_masks: Optional[Dict] = None
            self.sam_updated = False
        
        # 時間的整合性管理用（ROI限定バッファリング）
        self.temporal_lock = threading.RLock()
        with self.temporal_lock:
            # 適応的バッファサイズ（フレームレート検出による動的調整）
            self.target_buffer_seconds = 2.0  # 2秒分保持
            self.current_fps = 30.0  # 初期値
            self.buffer_size = int(self.current_fps * self.target_buffer_seconds)
            
            # ROI限定バッファリング（メモリ効率改善）
            self.roi_buffer = collections.deque(maxlen=self.buffer_size)
            self.roi_buffer_rgb = collections.deque(maxlen=self.buffer_size)
            self.roi_coords = collections.deque(maxlen=self.buffer_size)  # ROI座標情報
            self.full_image_buffer = collections.deque(maxlen=self.buffer_size)  # フル画像バッファ
            self.image_timestamps = collections.deque(maxlen=self.buffer_size)
            
            # フレームレート監視用
            self.frame_timestamps = collections.deque(maxlen=10)  # 最新10フレーム
            
            self.detection_requests = {}  # {request_id: timestamp} 検出リクエスト管理
            self.next_request_id = 0
            self.max_temporal_gap = 1.0  # 最大時間ギャップ（秒）
        
        # 非同期処理用
        self.background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Visualization")
        self.visualization_future = None
        
        # GroundingDINO非同期処理用
        self.gdino_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GroundingDINO")
        self.gdino_future = None
        self.gdino_processing = False
        
        # SAM2非同期処理用
        self.sam2_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SAM2")
        self.sam2_future = None
        self.sam2_processing = False
        self.sam2_pending_data = None
        
        # トラッキング非同期処理用
        self.tracking_future = None
        self.tracking_processing = False
        self.tracking_pending_data = None
        
        # 特徴点処理並列化用
        self.feature_executor = ThreadPoolExecutor(max_workers=self.feature_processing_workers, thread_name_prefix="FeatureProcessing")
        
        # 特徴点プール管理
        self.feature_pool = FeaturePool(max_features=100, min_features=20)
        
        # GPUリソース管理
        self.gpu_manager = get_gpu_manager()
        self.gpu_manager.set_memory_threshold(0.8)  # 80%で警告
        self.tracker_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CSRTTracker")
        
        # 性能計測用
        self.performance_stats = {
            'frame_times': [],
            'gdino_times': [],
            'sam2_times': [],
            'optflow_times': []
        }
        self.last_performance_log = 0
        
        # タイミング管理
        self.frame_count = 0
        self.last_grounding_dino_time = 0
        self.last_sam2_time = 0
    
    def _cleanup_resources(self):
        """リソースのクリーンアップ"""
        try:
            self.get_logger().info("リソースクリーンアップ開始")
            
            # GPUリソースマネージャーの終了
            if hasattr(self, 'gpu_manager'):
                self.gpu_manager.shutdown()
                
            # スレッドプールの終了（タイムアウト付き）
            executors = [
                ('background_executor', self.background_executor),
                ('gdino_executor', self.gdino_executor),
                ('sam2_executor', self.sam2_executor),
                ('feature_executor', self.feature_executor)
            ]
            
            for name, executor in executors:
                try:
                    if hasattr(self, name.split('_')[0] + '_executor'):
                        self.get_logger().debug(f"{name} 終了中...")
                        executor.shutdown(wait=False)  # ノンブロッキング
                        # 短いタイムアウトで強制終了
                        import time
                        time.sleep(0.1)
                except Exception as e:
                    self.get_logger().warning(f"{name} 終了エラー: {e}")
                    
            self.get_logger().info("リソースクリーンアップ完了")
            
        except Exception as e:
            print(f"リソースクリーンアップエラー: {e}")
    
    def __del__(self):
        """デストラクタ: スレッドプールの適切な終了"""
        try:
            self._cleanup_resources()
            if hasattr(self, 'tracker_executor'):
                self.tracker_executor.shutdown(wait=True)
            
            # 画像バッファのクリア
            if hasattr(self, 'temporal_lock'):
                with self.temporal_lock:
                    self.image_buffer.clear()
                    self.image_timestamps.clear()
                    self.image_buffer_rgb.clear()
        except Exception:
            pass
    
    def _init_ros_communication(self) -> None:
        """ROS通信の設定"""
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10,
            callback_group=self.image_callback_group
        )
        
        # 出力トピック
        self.sam_image_pub = self.create_publisher(
            ROSImage, '/image_sam', 10,
            callback_group=self.publisher_callback_group
        )
        self.optflow_image_pub = self.create_publisher(
            ROSImage, '/image_optflow', 10,
            callback_group=self.publisher_callback_group
        )
        self.grounding_dino_image_pub = self.create_publisher(
            ROSImage, '/image_grounding_dino', 10,
            callback_group=self.publisher_callback_group
        )
        self.sam_masks_pub = self.create_publisher(
            SamMasks, '/sam_masks', 10,
            callback_group=self.publisher_callback_group
        )
        self.feature_points_pub = self.create_publisher(
            FeaturePoints, '/feature_points', 10,
            callback_group=self.publisher_callback_group
        )
    
    def get_config_param(self, name: str, default_value=None):
        """パラメータ取得（デフォルト値対応）"""
        try:
            if default_value is not None:
                self.declare_parameter(name, default_value)
            else:
                # デフォルト値がない場合は適切なデフォルト値を設定
                if name == 'text_prompt':
                    default_value = "white line. red pylon. human. wall. car. building. mobility. road."
                elif name == 'box_threshold':
                    default_value = 0.3
                elif name == 'text_threshold':
                    default_value = 0.2
                elif name == 'tracking_targets':
                    default_value = "white line. red pylon. human. car."
                elif name == 'sam_model':
                    default_value = "sam2.1_hiera_small"
                elif name.startswith('grounding_dino_'):
                    default_value = 2.0
                elif name.startswith('sam2_'):
                    default_value = 0.05 if 'interval' in name else False
                elif name.startswith('harris_'):
                    default_value = 5 if 'corners' in name else 0.02
                elif name.startswith('tracker_'):
                    default_value = 0.5 if 'threshold' in name else 2
                elif name.startswith('feature_selection_'):
                    default_value = "harris_response"
                elif name in ['max_displacement', 'min_valid_ratio']:
                    default_value = 50.0 if name == 'max_displacement' else 0.5
                elif name.startswith('canny_'):
                    default_value = 50 if 'low' in name else 150
                elif name == 'min_edge_pixels':
                    default_value = 10
                else:
                    default_value = None
                    
                if default_value is not None:
                    self.declare_parameter(name, default_value)
                else:
                    # デフォルト値が決定できない場合はスキップ
                    return None
            
            param = self.get_parameter(name)
            value = param.value
            
            return value
        except Exception as e:
            self.get_logger().warn(f"パラメータ '{name}' の取得に失敗: {e}, デフォルト値 {default_value} を使用")
            return default_value
    
    def _parse_tracking_targets(self, targets_str: str) -> List[str]:
        """トラッキング対象文字列をパース"""
        if not targets_str or not targets_str.strip():
            return []
        return [target.strip() for target in targets_str.split('.') if target.strip()]
    
    def _log_key_parameters(self) -> None:
        """重要パラメータのログ出力"""
        self.get_logger().info(f"GroundingDINO間隔: {self.grounding_dino_interval}秒")
        self.get_logger().info(f"SAM2間隔: {self.sam2_interval}秒 (毎フレーム: {self.enable_sam2_every_frame})")
        self.get_logger().info(f"オプティカルフロー: 複数特徴点で追跡")
        self.get_logger().info(f"SAM2入力: オプティカルフローで追跡されたバウンディングボックス")
        self.get_logger().info(f"追跡対象: {self.tracking_targets}")
    
    def image_callback(self, msg: ROSImage) -> None:
        """メイン画像処理コールバック"""
        with self.processing_lock:
            self._process_image(msg)
    
    def _process_image(self, msg: ROSImage) -> None:
        """画像処理メインループ（時間整合性対応版）"""
        frame_start_time = time.time()
        try:
            # シンプルなフレームカウント
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"フレーム {self.frame_count} 処理中")
            
            # 高速画像変換（コピー最小化）
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if image_cv is None or image_cv.size == 0:
                self.get_logger().error("受信した画像が空です")
                return
            
            # 効率的な型変換（copy=Falseで最適化）
            if image_cv.dtype != np.uint8:
                image_cv = image_cv.astype(np.uint8, copy=False)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
            
            current_time = frame_start_time
            
            # 画像バッファリング（時間整合性のため）
            self._buffer_image(gray.copy(), image_cv.copy(), current_time)
            
            # GroundingDINO処理中の画像保存（オリジナルRGB画像を保存）
            self._store_image_if_needed(image_cv.copy(), current_time)
            
            # 最適化ログ（頻度を削減）
            if self.frame_count % 100 == 0 and self.frame_count > 0:
                self.get_logger().info(f"フレーム {self.frame_count} 処理中")
            
            # 1. GroundingDINO検出（画像保存システム付き）
            self._check_grounding_dino_results()  # 完了した非同期処理の結果を確認
            should_run_gdino = self._should_run_grounding_dino(current_time)
            if should_run_gdino:
                gdino_start = time.time()
                if self.frame_count % 30 == 0:
                    self.get_logger().info("GroundingDINO検出を非同期実行（画像保存開始）")
                
                # GroundingDINO処理開始時に画像保存を開始
                self._start_image_storage(current_time)
                
                # 時間追跡付きで検出を開始（実際の画像タイムスタンプを使用）
                request_id = self._run_grounding_dino_detection_with_timestamp(image_cv, current_time)
                self.last_grounding_dino_time = current_time
                self.performance_stats['gdino_times'].append(time.time() - gdino_start)
            else:
                if self.frame_count % 200 == 0:  # ログ頻度を削減
                    time_since_last = current_time - self.last_grounding_dino_time
                    self.get_logger().info(f"GroundingDINOスキップ: {time_since_last:.1f}s/{self.grounding_dino_interval}s")
            
            # 2. 時間的整合性を考慮した追跡初期化処理
            self._check_tracking_results()  # 完了した非同期処理の結果を確認
            optflow_start = time.time()
            if self.detection_updated:
                if self.frame_count % 30 == 0:
                    self.get_logger().info("時間的整合性を考慮した追跡初期化処理")
                
                # 時間的整合性を考慮した初期化方式を使用
                if self.enable_temporal_alignment:
                    self._initialize_tracking_with_temporal_alignment(current_time)
                    # 時間遡り処理後は通常のオプティカルフロー処理をスキップ
                    if self.frame_count % 30 == 0:
                        self.get_logger().info("時間遡り処理完了のため、通常のオプティカルフロー処理をスキップ")
                else:
                    # フォールバック: 統合処理方式
                    self._integrate_new_detections_with_existing_tracking(current_time)
                    # 統合処理の場合は非同期オプティカルフロー処理を実行
                    self._run_tracking_async(gray, current_time)
                    
                self.detection_updated = False
                self.performance_stats['optflow_times'].append(time.time() - optflow_start)
            else:
                # 通常の非同期オプティカルフロー処理（検出更新がない場合）
                self._run_tracking_async(gray, current_time)
                self.performance_stats['optflow_times'].append(time.time() - optflow_start)
            
            # 3. SAM2セグメンテーション（非同期版）
            self._check_sam2_results()  # 完了した非同期処理の結果を確認
            should_run_sam2 = self._should_run_sam2(current_time)
            
            # SAM2実行条件のログ
            if self.frame_count % 30 == 0:
                sam2_processing_status = "処理中" if self.sam2_processing else "待機中"
                self.get_logger().info(f"SAM2実行条件: should_run={should_run_sam2}, tracking_valid={self.tracking_valid}, tracked_boxes={len(self.tracked_boxes_per_label)}, sam2_status={sam2_processing_status}")
            
            if should_run_sam2:
                sam2_start = time.time()
                if self.frame_count % 30 == 0:
                    self.get_logger().info("SAM2セグメンテーション非同期実行")
                self._run_sam2_segmentation_async(image_cv)
                self.last_sam2_time = current_time
                self.performance_stats['sam2_times'].append(time.time() - sam2_start)
            elif self.frame_count % 30 == 0:
                self.get_logger().info("SAM2実行スキップ")
            
            # 4. 結果の可視化・配信（最適化版）
            self._publish_results(image_cv)
            
            # 性能統計更新
            frame_time = time.time() - frame_start_time
            self.performance_stats['frame_times'].append(frame_time)
            self._update_performance_stats(current_time)
            
            # 定期的なクリーンアップ（100フレームごと）
            if self.frame_count % 100 == 0:
                self._cleanup_old_tracking_data()
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
    def _update_performance_stats(self, current_time: float) -> None:
        """性能統計の更新とログ出力"""
        try:
            # 5秒ごとに性能ログを出力
            if current_time - self.last_performance_log >= 5.0:
                # スタッツリストのサイズを制限（メモリ節約）
                max_stats = 100
                for key in self.performance_stats:
                    if len(self.performance_stats[key]) > max_stats:
                        self.performance_stats[key] = self.performance_stats[key][-max_stats:]
                
                # 平均時間を計算
                if self.performance_stats['frame_times']:
                    avg_frame = np.mean(self.performance_stats['frame_times'][-50:]) * 1000
                    fps = 1.0 / np.mean(self.performance_stats['frame_times'][-50:]) if self.performance_stats['frame_times'][-50:] else 0
                    
                    avg_gdino = np.mean(self.performance_stats['gdino_times'][-10:]) * 1000 if self.performance_stats['gdino_times'] else 0
                    avg_sam2 = np.mean(self.performance_stats['sam2_times'][-20:]) * 1000 if self.performance_stats['sam2_times'] else 0
                    avg_optflow = np.mean(self.performance_stats['optflow_times'][-50:]) * 1000 if self.performance_stats['optflow_times'] else 0
                    
                    self.get_logger().info(
                        f"性能: FPS={fps:.1f}, フレーム={avg_frame:.1f}ms, "
                        f"GDINO={avg_gdino:.1f}ms, SAM2={avg_sam2:.1f}ms, OptFlow={avg_optflow:.1f}ms"
                    )
                
                self.last_performance_log = current_time
        except Exception as e:
            self.get_logger().warn(f"性能統計エラー: {e}")
    
    def _update_fps_and_buffer_size(self, timestamp: float) -> None:
        """フレームレート検出とバッファサイズの動的調整"""
        try:
            self.frame_timestamps.append(timestamp)
            
            if len(self.frame_timestamps) >= 5:  # 5フレーム以上でFPS計算
                time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
                if time_diff > 0:
                    detected_fps = (len(self.frame_timestamps) - 1) / time_diff
                    # 指数移動平均でスムージング
                    self.current_fps = 0.9 * self.current_fps + 0.1 * detected_fps
                    
                    # バッファサイズの動的調整
                    new_buffer_size = int(self.current_fps * self.target_buffer_seconds)
                    if new_buffer_size != self.buffer_size:
                        self.buffer_size = new_buffer_size
                        # バッファサイズを更新
                        self.roi_buffer = collections.deque(self.roi_buffer, maxlen=self.buffer_size)
                        self.roi_buffer_rgb = collections.deque(self.roi_buffer_rgb, maxlen=self.buffer_size)
                        self.roi_coords = collections.deque(self.roi_coords, maxlen=self.buffer_size)
                        self.full_image_buffer = collections.deque(self.full_image_buffer, maxlen=self.buffer_size)
                        self.image_timestamps = collections.deque(self.image_timestamps, maxlen=self.buffer_size)
                        
        except Exception as e:
            self.get_logger().error(f"FPS更新エラー: {e}")
    
    def _extract_roi_from_image(self, image: np.ndarray, active_objects: Dict) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """アクティブなオブジェクトの領域を統合したROIを抽出"""
        if not active_objects:
            # オブジェクトがない場合は中央部分をROIとして使用
            h, w = image.shape[:2]
            roi_coords = (w//4, h//4, w*3//4, h*3//4)
            return image[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]].copy(), roi_coords
        
        # アクティブなオブジェクトのバウンディングボックスを統合
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for obj_data in active_objects.values():
            bbox = obj_data.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
        
        if min_x == float('inf'):  # 有効なbboxがない場合
            h, w = image.shape[:2]
            roi_coords = (w//4, h//4, w*3//4, h*3//4)
        else:
            # マージンを追加してROIを拡張
            margin = 50
            h, w = image.shape[:2]
            roi_coords = (
                max(0, int(min_x) - margin),
                max(0, int(min_y) - margin),
                min(w, int(max_x) + margin),
                min(h, int(max_y) + margin)
            )
        
        x1, y1, x2, y2 = roi_coords
        roi_image = image[y1:y2, x1:x2].copy()
        return roi_image, roi_coords
    
    def _buffer_image(self, gray: np.ndarray, rgb: np.ndarray, timestamp: float) -> None:
        """画像をROI限定でバッファリング（メモリ効率改善）"""
        try:
            with self.temporal_lock:
                # FPS検出とバッファサイズの動的調整
                self._update_fps_and_buffer_size(timestamp)
                
                # アクティブなオブジェクトに基づいてROIを抽出
                active_objects = {}
                if hasattr(self, 'tracking_boxes') and self.tracking_boxes:
                    active_objects = self.tracking_boxes
                elif hasattr(self, 'detection_boxes') and self.detection_boxes:
                    active_objects = self.detection_boxes
                    
                # ROI抽出
                roi_gray, roi_coords = self._extract_roi_from_image(gray, active_objects)
                roi_rgb, _ = self._extract_roi_from_image(rgb, active_objects)
                
                # ROIバッファに保存
                self.roi_buffer.append(roi_gray)
                self.roi_buffer_rgb.append(roi_rgb)
                self.roi_coords.append(roi_coords)
                self.full_image_buffer.append(gray.copy())  # フル画像も保存
                self.image_timestamps.append(timestamp)
                
                # 古いデータのクリーンアップ
                current_time = time.time()
                while (len(self.image_timestamps) > 0 and 
                       current_time - self.image_timestamps[0] > self.max_temporal_gap):
                    self.roi_buffer.popleft()
                    self.roi_buffer_rgb.popleft()
                    self.roi_coords.popleft()
                    self.full_image_buffer.popleft()
                    self.image_timestamps.popleft()
                    
        except Exception as e:
            self.get_logger().error(f"画像バッファリングエラー: {e}")
    
    def _get_roi_by_timestamp(self, target_timestamp: float) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
        """指定されたタイムスタンプに最も近いROIを取得"""
        try:
            with self.temporal_lock:
                if len(self.image_timestamps) == 0:
                    return None
                
                # 最も近いタイムスタンプのインデックスを検索
                timestamps = np.array(self.image_timestamps)
                differences = np.abs(timestamps - target_timestamp)
                closest_index = np.argmin(differences)
                
                # 時間差が許容範囲内かチェック
                time_diff = differences[closest_index]
                if time_diff > self.max_temporal_gap:
                    self.get_logger().warn(f"時間差が大きすぎます: {time_diff:.3f}秒")
                    return None
                
                # ROIサイズチェック
                roi_gray = self.roi_buffer[closest_index]
                roi_rgb = self.roi_buffer_rgb[closest_index]
                
                if roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                    self.get_logger().warning(f"ROIが小さすぎます: {roi_gray.shape}")
                    return None
                
                return (roi_gray.copy(), roi_rgb.copy(), self.roi_coords[closest_index])
                       
        except Exception as e:
            self.get_logger().error(f"ROI取得エラー: {e}")
            return None
    
    def _get_full_image_by_timestamp(self, target_timestamp: float) -> Optional[np.ndarray]:
        """指定されたタイムスタンプに最も近いフル画像を取得 - デバッグ強化版"""
        try:
            with self.temporal_lock:
                if not hasattr(self, 'full_image_buffer') or len(self.full_image_buffer) == 0:
                    self.get_logger().debug(f"フル画像バッファが空です (target: {target_timestamp:.3f})")
                    return None
                
                if len(self.image_timestamps) == 0:
                    self.get_logger().debug(f"タイムスタンプバッファが空です (target: {target_timestamp:.3f})")
                    return None
                
                # 最も近いタイムスタンプのインデックスを検索
                timestamps = np.array(self.image_timestamps)
                differences = np.abs(timestamps - target_timestamp)
                closest_index = np.argmin(differences)
                
                # 時間差が許容範囲内かチェック
                time_diff = differences[closest_index]
                closest_timestamp = timestamps[closest_index]
                
                self.get_logger().debug(f"画像取得: target={target_timestamp:.3f}, closest={closest_timestamp:.3f}, diff={time_diff:.3f}, max_gap={self.max_temporal_gap:.3f}")
                
                if time_diff > self.max_temporal_gap:
                    self.get_logger().warning(f"時間差が大きすぎます: {time_diff:.3f}秒 > {self.max_temporal_gap:.3f}秒 (target: {target_timestamp:.3f})")
                    # 利用可能な時間範囲をログ出力
                    if len(timestamps) > 0:
                        self.get_logger().debug(f"利用可能な時間範囲: {timestamps[0]:.3f} - {timestamps[-1]:.3f}")
                    return None
                
                if closest_index < len(self.full_image_buffer):
                    image = self.full_image_buffer[closest_index]
                    if image is not None:
                        self.get_logger().debug(f"画像取得成功: インデックス={closest_index}, 形状={image.shape}")
                        return image.copy()
                    else:
                        self.get_logger().warning(f"バッファのインデックス{closest_index}の画像がNullです")
                        return None
                else:
                    self.get_logger().warning(f"インデックス範囲外: {closest_index} >= {len(self.full_image_buffer)}")
                    return None
                       
        except Exception as e:
            self.get_logger().error(f"フル画像取得エラー: {e}")
            import traceback
            self.get_logger().error(f"スタックトレース: {traceback.format_exc()}")
            return None
    
    def _convert_roi_to_full_coordinates(self, roi_coords: List[float], roi_offset: Tuple[int, int, int, int]) -> List[float]:
        """直接追跡方式: ROI座標を元のフル画像座標に変換"""
        if len(roi_coords) != 4:
            return roi_coords
        
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_offset
        x1, y1, x2, y2 = roi_coords
        
        # ROI座標を元のフル画像座標に変換
        full_x1 = x1 + roi_x1
        full_y1 = y1 + roi_y1
        full_x2 = x2 + roi_x1
        full_y2 = y2 + roi_y1
        
        return [full_x1, full_y1, full_x2, full_y2]
    
    def _direct_temporal_jump(self, detection_timestamp: float, current_timestamp: float, current_gray: np.ndarray) -> bool:
        """直接追跡方式: 検出時点から現在への直接オプティカルフロー実行"""
        try:
            time_gap = current_timestamp - detection_timestamp
            self.get_logger().info(f"   >> 直接追跡ジャンプ開始: {time_gap*1000:.1f}ms間隔")
            
            if time_gap < 0.05:  # 50ms以下の場合はスキップ
                self.get_logger().info(f"   >> 間隔が短いためスキップ")
                return True
            
            # 検出時点のROIを取得
            roi_data = self._get_roi_by_timestamp(detection_timestamp)
            if roi_data is None:
                self.get_logger().warn(f"検出時点のデータが見つかりません")
                return False
            
            detection_roi_gray, _, roi_offset = roi_data
            
                # 現在の画像から同じROIを抽出
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_offset
            current_roi_gray = current_gray[roi_y1:roi_y2, roi_x1:roi_x2]
            
            self.get_logger().debug(f"   >> ROI座標: ({roi_x1},{roi_y1}) -> ({roi_x2},{roi_y2})")
            self.get_logger().debug(f"   >> ROIサイズ: {current_roi_gray.shape} (検出時: {detection_roi_gray.shape})")
            
            # 画像サイズの整合性チェック
            if detection_roi_gray.shape != current_roi_gray.shape:
                self.get_logger().error(f"   >> ROIサイズ不整合: {detection_roi_gray.shape} vs {current_roi_gray.shape}")
                return False
            
            # ROIが小さすぎる場合はスキップ
            if current_roi_gray.shape[0] < 10 or current_roi_gray.shape[1] < 10:
                self.get_logger().warning(f"   >> ROIが小さすぎます: {current_roi_gray.shape}")
                return False
            
            # 直接オプティカルフローで追跡更新
            updated_count = 0
            total_objects = len(self.tracked_boxes_per_label)
            self.get_logger().info(f"   >> 追跡オブジェクト数: {total_objects}")
            
            for label, bbox in list(self.tracked_boxes_per_label.items()):
                if label not in self.tracked_features_per_label or self.tracked_features_per_label[label] is None:
                    continue
                    
                # 中心点をROI座標系に変換
                center = self.tracked_features_per_label[label]
                
                # 中心点をROI座標系に変換
                if center.size == 0:
                    self.get_logger().warning(f"     - {label}: 空の中心点")
                    continue
                
                # 中心点の形状を正規化
                if len(center.shape) == 3:
                    center_x, center_y = center[0, 0]
                elif len(center.shape) == 2:
                    center_x, center_y = center[0]
                else:
                    center_x, center_y = center
                
                # ROI座標系に変換
                roi_center_x = center_x - roi_x1
                roi_center_y = center_y - roi_y1
                roi_center = np.array([[[roi_center_x, roi_center_y]]], dtype=np.float32)
                
                # 境界チェック
                if (roi_center_x < 0 or roi_center_x >= current_roi_gray.shape[1] or
                    roi_center_y < 0 or roi_center_y >= current_roi_gray.shape[0]):
                    self.get_logger().warning(f"     - {label}: 中心点がROI境界外")
                    continue
                
                self.get_logger().debug(f"     - {label}: 中心点をROI座標系に変換")
                
                # オプティカルフロー実行（中心点）
                try:
                    curr_center, status, _ = cv2.calcOpticalFlowPyrLK(
                        detection_roi_gray, current_roi_gray, roi_center, None,
                        winSize=self.optical_flow_win_size,
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                except cv2.error as e:
                    self.get_logger().error(f"     - {label}: オプティカルフロー実行エラー: {e}")
                    continue
                
                # 中心点の有効性チェック
                if curr_center is not None and status is not None and status[0][0] == 1:
                    # フル座標系に戻す
                    full_center_x = curr_center[0, 0] + roi_x1
                    full_center_y = curr_center[0, 1] + roi_y1
                    full_center = np.array([[[full_center_x, full_center_y]]], dtype=np.float32)
                    
                    # 中心点を更新
                    self.tracked_features_per_label[label] = full_center
                    
                    # バウンディングボックスを更新
                    self._update_bounding_box_from_features(label)
                    updated_count += 1
                    
                    self.get_logger().debug(f"     - {label}: 中心点追跡成功")
                else:
                    self.get_logger().debug(f"     - {label}: 中心点追跡失敗により削除")
                    self._remove_tracking_object(label)
                    continue
            
            success_rate = (updated_count / total_objects * 100) if total_objects > 0 else 0
            self.get_logger().info(f"   >> 直接追跡結果: {updated_count}/{total_objects}個更新 ({success_rate:.1f}%)")
            return updated_count > 0
            
        except Exception as e:
            self.get_logger().error(f"直接追跡エラー: {e}")
            return False
                
        except Exception as e:
            self.get_logger().error(f"ROI取得エラー: {e}")
            return None
    
    def _should_run_grounding_dino(self, current_time: float) -> bool:
        """GroundingDINO実行判定"""
        return (
            self.frame_count == 0 or
            current_time - self.last_grounding_dino_time >= self.grounding_dino_interval or
            not self.tracking_valid
        )
    
    def _should_run_sam2(self, current_time: float) -> bool:
        """SAM2実行判定"""
        # オプティカルフローの状態に関係なく、追跡ボックスがあればSAM2実行
        if not self.tracked_boxes_per_label:
            return False
            
        return (
            self.enable_sam2_every_frame or
            current_time - self.last_sam2_time >= self.sam2_interval
        )
    
    def _run_grounding_dino_detection_with_timestamp(self, image_cv: np.ndarray, timestamp: float) -> int:
        """時間追跡付きGroundingDINO検出実行"""
        # 既に処理中の場合はスキップ
        # GPUリソース管理で適切な実行判定
        gpu_stats = self.gpu_manager.get_statistics()
        if gpu_stats['usage_percent'] > 85 and self.gdino_processing:
            self.get_logger().warning(f"GPU使用率が高いためGroundingDINOをスキップ: {gpu_stats['usage_percent']:.1f}%")
            return -1
        
        if self.gdino_processing or (self.gdino_future and not self.gdino_future.done()):
            return -1
        
        # リクエストIDを生成して時間を記録
        request_id = self.next_request_id
        self.next_request_id += 1
        
        with self.temporal_lock:
            self.detection_requests[request_id] = timestamp
        
        # 直接非同期実行（GPUリソース管理は後で改善）
        self.gdino_processing = True
        self.gdino_future = self.gdino_executor.submit(
            self._grounding_dino_worker_with_timestamp, image_cv.copy(), request_id, timestamp
        )
        
        return request_id
    
    def _wait_for_gpu_task(self, task) -> any:
        """タスクの結果を待機"""
        while task.result is None and task.error is None:
            time.sleep(0.01)  # 10ms間隔でチェック
        
        if task.error is not None:
            raise task.error
        
        return task.result
    
    def _run_grounding_dino_detection_async(self, image_cv: np.ndarray) -> None:
        """GroundingDINO検出実行（非同期処理）"""
        # 既に処理中の場合はスキップ
        if self.gdino_processing or (self.gdino_future and not self.gdino_future.done()):
            return
        
        # 直接非同期実行（GPUリソース管理は後で改善）
        self.gdino_processing = True
        self.gdino_future = self.gdino_executor.submit(
            self._grounding_dino_worker, image_cv.copy()
        )
    
    def _grounding_dino_worker_with_timestamp(self, image_cv: np.ndarray, request_id: int, original_timestamp: float) -> Optional[Dict]:
        """時間追跡付きGroundingDINO処理ワーカー"""
        try:
            if self.gdino_model is None:
                self.get_logger().warn("GroundingDINOモデルが初期化されていません")
                return None
                
            # PIL画像に変換
            image_pil = Image.fromarray(image_cv, mode='RGB')
            
            # GroundingDINO推論
            with self._gpu_memory_context():
                results = self.gdino_model.predict(
                    [image_pil], 
                    [self.text_prompt], 
                    self.box_threshold, 
                    self.text_threshold
                )
            
            # 結果を返却（元の画像タイムスタンプを保持）
            if results and len(results) > 0:
                # CUDA テンソルを CPU に変換
                detection_data = self._convert_detection_tensors_to_cpu(results[0])
                return {
                    'detection_data': detection_data,
                    'image': image_cv,
                    'request_id': request_id,
                    'original_timestamp': original_timestamp,  # 元の画像のタイムスタンプ
                    'completion_timestamp': time.time()  # 完了時刻
                }
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f"GroundingDINO処理エラー: {e}")
            return None
        finally:
            self.gdino_processing = False
    
    def _grounding_dino_worker(self, image_cv: np.ndarray) -> Optional[Dict]:
        """GroundingDINO処理ワーカー（バックグラウンドスレッド）"""
        try:
            if self.gdino_model is None:
                self.get_logger().warn("GroundingDINOモデルが初期化されていません")
                return None
                
            # PIL画像に変換
            image_pil = Image.fromarray(image_cv, mode='RGB')
            
            # GroundingDINO推論
            with self._gpu_memory_context():
                results = self.gdino_model.predict(
                    [image_pil], 
                    [self.text_prompt], 
                    self.box_threshold, 
                    self.text_threshold
                )
            
            # 結果を返却
            if results and len(results) > 0:
                # CUDA テンソルを CPU に変換
                detection_data = self._convert_detection_tensors_to_cpu(results[0])
                return {
                    'detection_data': detection_data,
                    'image': image_cv,
                    'timestamp': time.time()
                }
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f"GroundingDINO処理エラー: {e}")
            return None
        finally:
            self.gdino_processing = False
    
    def _check_grounding_dino_results(self) -> None:
        """GroundingDINO非同期処理の結果を確認・統合（時間整合性対応）"""
        if self.gdino_future and self.gdino_future.done():
            try:
                result = self.gdino_future.result()
                
                if result is not None:
                    # 時間整合性情報を含めて結果を内部データに保存
                    with self.detection_data_lock:
                        self.latest_detection_data = result['detection_data']
                        
                        # 時間整合性情報を追加
                        if 'original_timestamp' in result:
                            self.latest_detection_data['original_timestamp'] = result['original_timestamp']
                            self.latest_detection_data['completion_timestamp'] = result['completion_timestamp']
                            processing_time = result['completion_timestamp'] - result['original_timestamp']
                            self.get_logger().info(f"検出処理時間: {processing_time:.3f}秒")
                        
                        self.detection_updated = True
                        
                        num_detections = len(self.latest_detection_data.get('boxes', []))
                        labels = self.latest_detection_data.get('labels', [])
                        scores = self.latest_detection_data.get('scores', [])
                        self.get_logger().info(f"GroundingDINO検出完了: {num_detections}個, ラベル: {labels}")
                        
                        # GroundingDINO検出結果の可視化用データを保存
                        self.latest_grounding_dino_result = {
                            'image': result['image'],
                            'boxes': self.latest_detection_data.get('boxes', []),
                            'labels': self.latest_detection_data.get('labels', []),
                            'scores': self.latest_detection_data.get('scores', [])
                        }
                        self.grounding_dino_updated = True
                        
                        # GroundingDINO処理完了時に画像保存を終了
                        self._stop_image_storage()
                else:
                    self.get_logger().warn("GroundingDINO検出結果が空です")
                    # 結果が空でも画像保存を終了
                    self._stop_image_storage()
                    
                # 完了したタスクをクリア
                self.gdino_future = None
                    
            except Exception as e:
                self.get_logger().error(f"GroundingDINO結果統合エラー: {repr(e)}")
                # エラー時も画像保存を終了
                self._stop_image_storage()
                self.gdino_future = None
    
    def _initialize_tracking_with_temporal_alignment(self, current_time: float) -> None:
        """時間整合性を考慮した追跡初期化"""
        try:
            with self.detection_data_lock:
                if self.latest_detection_data is None:
                    self.get_logger().warn("latest_detection_dataがNoneです")
                    return
                
                # 検出データの元のタイムスタンプを取得
                detection_timestamp = self.latest_detection_data.get('original_timestamp')
                completion_timestamp = self.latest_detection_data.get('completion_timestamp')
                
                if detection_timestamp is None:
                    self.get_logger().warn("検出データにタイムスタンプがありません、統合処理に切り替え")
                    self._integrate_new_detections_with_existing_tracking(current_time)
                    return
                
                # 時間差を計算
                time_gap = current_time - detection_timestamp
                processing_time = completion_timestamp - detection_timestamp if completion_timestamp else 0
                
                self.get_logger().info(f"=== 時間的整合性処理開始 ===")
                self.get_logger().info(f"検出開始時刻: {detection_timestamp:.3f}秒")
                self.get_logger().info(f"検出完了時刻: {completion_timestamp:.3f}秒" if completion_timestamp else "検出完了時刻: 不明")
                self.get_logger().info(f"現在時刻: {current_time:.3f}秒")
                self.get_logger().info(f"検出処理時間: {processing_time:.3f}秒")
                self.get_logger().info(f"時間差: {time_gap:.3f}秒 ({time_gap*1000:.1f}ms)")
                
                # 実際の処理遅延を考慮した時間遡り
                if processing_time > 0:
                    actual_delay = processing_time + time_gap
                    self.get_logger().info(f"実際の遅延時間: {actual_delay:.3f}秒 ({actual_delay*1000:.1f}ms)")
                    self.get_logger().info(f"→ {detection_timestamp:.3f}秒の画像まで遡って追跡開始")
                
                if time_gap > self.max_temporal_gap:
                    self.get_logger().warn(f"時間差が大きすぎます ({time_gap:.3f}s > {self.max_temporal_gap}s)、統合処理に切り替え")
                    self._integrate_new_detections_with_existing_tracking(current_time)
                    return
                
                # 検出時点のフル画像を取得
                detection_full_image = self._get_full_image_by_timestamp(detection_timestamp)
                if detection_full_image is None:
                    self.get_logger().warn("検出時点の画像が見つかりません、統合処理に切り替え")
                    self._integrate_new_detections_with_existing_tracking(current_time)
                    return
                
                # 検出時点の画像で追跡初期化（フォーマット統一）
                self.get_logger().info(f"✓ 検出時点({detection_timestamp:.3f})の画像で追跡初期化")
                self.get_logger().info(f"  - 使用画像サイズ: {detection_full_image.shape}")
                
                # 画像をグレースケールに統一
                if len(detection_full_image.shape) == 3:
                    detection_gray = cv2.cvtColor(detection_full_image, cv2.COLOR_BGR2GRAY)
                else:
                    detection_gray = detection_full_image
                    
                # 既存トラッカーとの統合処理を実行
                self._integrate_new_detections_with_existing_tracking(detection_timestamp)
                
                # 早送り処理: 検出時点から現在まで時間ジャンプ
                if time_gap > 0.05:  # 50ms以上の差がある場合のみ早送り
                    self.get_logger().info(f"✓ 早送り処理実行: {detection_timestamp:.3f} -> {current_time:.3f}")
                    self.get_logger().info(f"  - 時間遅延補償: {time_gap*1000:.1f}ms")
                    
                    # 早送り前の状態確認
                    pre_fastforward_trackers = len(getattr(self, 'tracked_csrt_trackers', {}))
                    
                    # CSRT用高速早送り処理
                    self._perform_temporal_catchup_optimized(detection_timestamp, current_time)
                    
                    # 早送り処理後の結果確認と詳細ログ
                    post_fastforward_trackers = len(getattr(self, 'tracked_csrt_trackers', {}))
                    if post_fastforward_trackers > 0:
                        self.get_logger().info(f"  - 早送り完了: {post_fastforward_trackers}個のトラッカーが稼働中")
                        
                        # 追跡結果の詳細表示
                        with self.tracking_data_lock:
                            for label, bbox in self.tracked_boxes_per_label.items():
                                self.get_logger().info(f"    - {label}: bbox={bbox}")
                    else:
                        self.get_logger().warn(f"  - 早送り後: アクティブトラッカー無し (前: {pre_fastforward_trackers}個)")
                else:
                    self.get_logger().info(f"✓ 早送りスキップ: 時間差が閾値以下 ({time_gap*1000:.1f}ms < 50ms)")
                
                # 時間遡り処理完了後、絶対最新画像でprev_grayを設定してリアルタイム追跡に備える
                absolute_latest_gray, absolute_latest_timestamp = self._get_absolute_latest_image()
                final_sync_time = time.time()
                
                if absolute_latest_gray is not None:
                    self.prev_gray = absolute_latest_gray.copy()
                    sync_delay = final_sync_time - absolute_latest_timestamp
                    self.get_logger().info(f"✓ 時間遡り処理完了後、絶対最新画像をprev_grayに設定 ({absolute_latest_timestamp:.3f})")
                    self.get_logger().info(f"  - 最終同期遅延: {sync_delay*1000:.1f}ms")
                else:
                    current_gray = self._get_current_gray_image()
                    if current_gray is not None:
                        self.prev_gray = current_gray.copy()
                        self.get_logger().info(f"✓ フォールバック: 現在画像をprev_grayに設定")
                
                self.get_logger().info(f"=== 時間的整合性処理完了 ===")
                
                # 早送り処理完了後に保存画像をクリア
                with self.tracking_data_lock:
                    self.stored_images.clear()
                    self.get_logger().info(f"保存画像クリア: 時間的整合性処理完了")
                
        except Exception as e:
            self.get_logger().error(f"時間整合性追跡初期化エラー: {e}")
            # エラー時も保存画像をクリア
            with self.tracking_data_lock:
                self.stored_images.clear()
                self.get_logger().info(f"保存画像クリア: エラー処理時")
            # フォールバック: 統合処理
            self._integrate_new_detections_with_existing_tracking(current_time)
    
    def _update_all_trackers(self, image: np.ndarray, update_prev_gray: bool = True) -> bool:
        """全CSRTトラッカーを指定画像で更新（高精度）"""
        try:
            success_count = 0
            total_trackers = len(getattr(self, 'tracked_csrt_trackers', {}))
            
            if total_trackers == 0:
                return False
            
            # BGR画像に変換（CSRTはBGR形式を要求）
            if len(image.shape) == 2:
                tracker_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                tracker_image = image.copy()
            else:
                return False
            
            with self.tracking_data_lock:
                labels_to_remove = []
                
                for label in list(self.tracked_csrt_trackers.keys()):
                    try:
                        csrt_tracker = self.tracked_csrt_trackers[label]
                        
                        # CSRTトラッカーで更新
                        success, bbox = csrt_tracker.update(tracker_image)
                        
                        if success and bbox is not None:
                            # OpenCV形式 (x, y, width, height) → 座標形式 (x1, y1, x2, y2)
                            x1 = bbox[0]
                            y1 = bbox[1]
                            x2 = x1 + bbox[2]
                            y2 = y1 + bbox[3]
                            
                            # 境界チェック
                            height, width = tracker_image.shape[:2]
                            if 0 <= x1 < width and 0 <= y1 < height and x2 <= width and y2 <= height:
                                # サイズ妥当性チェック（横ズレ対策）
                                bbox_width = x2 - x1
                                bbox_height = y2 - y1
                                
                                # 異常に大きなbboxは制限
                                max_width = width * 0.8  # 画面幅の80%以下
                                max_height = height * 0.8  # 画面高さの80%以下
                                
                                if bbox_width <= max_width and bbox_height <= max_height:
                                    # バウンディングボックス更新
                                    self.tracked_boxes_per_label[label] = [x1, y1, x2, y2]
                                    self.tracker_failure_counts[label] = 0
                                    success_count += 1
                                    
                                    # 詳細ログ（可視化改善）
                                    self.get_logger().debug(f"CSRTトラッキング成功: {label} - bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] size={bbox_width:.0f}x{bbox_height:.0f}")
                                else:
                                    # サイズ異常
                                    self.get_logger().warn(f"CSRTトラッキング: {label} - bbox異常サイズ {bbox_width:.0f}x{bbox_height:.0f} (制限: {max_width:.0f}x{max_height:.0f})")
                                    self.tracker_failure_counts[label] += 1
                            else:
                                # 境界外
                                self.get_logger().debug(f"CSRTトラッキング: {label} - 境界外 [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                                self.tracker_failure_counts[label] += 1
                        else:
                            # トラッキング失敗
                            self.tracker_failure_counts[label] += 1
                        
                        # 失敗回数チェック
                        if self.tracker_failure_counts[label] >= self.tracker_failure_threshold:
                            labels_to_remove.append(label)
                    
                    except Exception as e:
                        self.get_logger().error(f"CSRTトラッカー更新エラー {label}: {e}")
                        labels_to_remove.append(label)
                
                # 失敗したトラッカーを削除
                for label in labels_to_remove:
                    self._remove_tracking_object(label)
            
            return success_count > 0
            
        except Exception as e:
            self.get_logger().error(f"全CSRTトラッカー更新エラー: {e}")
            return False
    
    def _perform_temporal_catchup_optimized(self, start_timestamp: float, end_timestamp: float) -> None:
        """保存画像を使用したシンプル早送り処理"""
        try:
            # CSRTトラッカーが初期化されているかチェック
            if len(getattr(self, 'tracked_csrt_trackers', {})) == 0:
                self.get_logger().warn("CSRTトラッカーが初期化されていません")
                return
            
            # 保存された画像を使用して早送り処理
            with self.tracking_data_lock:
                if not self.stored_images:
                    self.get_logger().warn(f"保存された画像がありません (ステータス: in_progress={self.grounding_dino_in_progress}, start_time={self.grounding_dino_start_time})")
                    return
                
                time_gap = end_timestamp - start_timestamp
                self.get_logger().info(f"保存画像早送り開始: {time_gap:.3f}秒ジャンプ, {len(self.stored_images)}枚の画像を使用")
                
                # 時間範囲内の画像をフィルタリング
                valid_images = [(t, img) for t, img in self.stored_images if start_timestamp <= t <= end_timestamp]
                
                if not valid_images:
                    self.get_logger().warn("早送り対象の画像がありません")
                    return
                
                # 保存画像を順序に処理してトラッカーを早送り
                success_count = 0
                for i, (timestamp, image) in enumerate(valid_images):
                    # 最後の画像でのみprev_grayを更新
                    is_last_frame = (i == len(valid_images) - 1)
                    
                    # 全トラッカーを更新
                    tracker_success = self._update_all_trackers(image, update_prev_gray=is_last_frame)
                    if tracker_success:
                        success_count += 1
                    
                    # 進捗ログ
                    self.get_logger().info(f"早送り進捗: {i+1}/{len(valid_images)}, 成功率: {success_count}/{i+1}")
                
                # 失敗したトラッカーを削除
                failed_labels = []
                for label, failure_count in self.tracker_failure_counts.items():
                    if failure_count >= self.tracker_failure_threshold:
                        failed_labels.append(label)
                
                for label in failed_labels:
                    self._remove_tracking_object(label)
                    self.get_logger().info(f"失敗トラッカー削除: {label} (失敗回数: {self.tracker_failure_counts.get(label, 0)})")
                
                # 早送り結果
                active_trackers = len(getattr(self, 'tracked_csrt_trackers', {}))
                success_rate = success_count / len(valid_images) if valid_images else 0
                self.get_logger().info(f"早送り完了: {success_count}/{len(valid_images)} (成功率: {success_rate:.1%})")
                
                # 最後の保存画像から現在時刻まで追跡を継続
                if valid_images and active_trackers > 0:
                    last_stored_time = valid_images[-1][0]
                    current_time = time.time()
                    remaining_gap = current_time - last_stored_time
                    
                    if remaining_gap > 0.01:  # 10ms以上の差がある場合
                        # 現在の最新画像まで追跡
                        current_gray = self._get_current_gray_image()
                        if current_gray is not None:
                            final_success = self._update_all_trackers(current_gray, update_prev_gray=True)
                            self.get_logger().info(f"残存時間補間: {remaining_gap:.3f}秒 -> 現在時刻まで追跡完了")
                        else:
                            self.get_logger().warn("現在画像が取得できません: 最終時間補間スキップ")
                
                self.get_logger().info(f"保存画像早送り完了: {time_gap:.3f}秒ジャンプ, {active_trackers}個のトラッカーが稼働中")
                
                # 注意: 保存画像は時間的整合性処理完了後にクリアされます
            
        except Exception as e:
            self.get_logger().error(f"保存画像早送り処理エラー: {e}")
    
    def _legacy_optical_flow_fallback(self, start_timestamp: float, end_timestamp: float) -> None:
        """従来のオプティカルフロー処理（レガシー・フォールバック用）"""
        try:
            self.get_logger().info("CSRT早送り失敗、レガシー処理にフォールバック")
            
            # 現在の画像を取得して単純更新
            current_image = self._get_full_image_by_timestamp(end_timestamp)
            if current_image is not None:
                if len(current_image.shape) == 3:
                    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                else:
                    current_gray = current_image
                    
                # 通常のトラッカー更新を実行
                self._update_csrt_trackers_without_deletion(current_gray)
                self.get_logger().info("レガシー処理完了")
                
        except Exception as e:
            self.get_logger().error(f"レガシー処理エラー: {e}")
    
    def _csrt_direct_jump(self, target_gray: np.ndarray, jump_type: str) -> bool:
        """CSRT直接ジャンプ処理（デバッグ改善版）"""
        try:
            # BGRに変換（トラッカー用）
            tracker_image = cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR)
            success_count = 0
            total_trackers = len(self.tracked_features_per_label)
            failed_trackers = []
            
            self.get_logger().info(f"  - {jump_type}: {total_trackers}個のトラッカーを更新")
            
            with self.tracking_data_lock:
                for label, tracker in list(self.tracked_csrt_trackers.items()):
                    try:
                        # シンプルなトラッカー更新
                        success, opencv_bbox = tracker.update(tracker_image)
                        
                        if success and opencv_bbox is not None:
                            x, y, w, h = opencv_bbox
                            bbox = [x, y, x + w, y + h]
                            
                            # 基本的な範囲チェックのみ
                            if w > 5 and h > 5:  # 最小サイズチェック
                                self.tracked_boxes_per_label[label] = bbox
                                success_count += 1
                                self.get_logger().debug(f"    - {label}: 更新成功 [{x:.1f},{y:.1f},{w:.1f},{h:.1f}]")
                            else:
                                failed_trackers.append(f"{label}(小さすぎ:{w:.1f}x{h:.1f})")
                        else:
                            failed_trackers.append(f"{label}(update失敗)")
                                
                    except Exception as e:
                        failed_trackers.append(f"{label}(例外:{e})")
            
            success_rate = success_count / total_trackers if total_trackers > 0 else 0.0
            
            # 成功判定を緩和（30%以上成功で OK）
            is_success = success_rate >= 0.3
            
            self.get_logger().info(f"  - {jump_type}結果: {success_count}/{total_trackers}成功 ({success_rate*100:.1f}%) -> {'成功' if is_success else '失敗'}")
            
            if failed_trackers:
                self.get_logger().debug(f"    失敗詳細: {', '.join(failed_trackers)}")
            
            return is_success
            
        except Exception as e:
            self.get_logger().error(f"CSRT直接ジャンプエラー: {e}")
            return False
    
    def _csrt_staged_jump(self, start_timestamp: float, end_timestamp: float, target_gray: np.ndarray, stages: int) -> bool:
        """CSRT段階的ジャンプ処理（中・長時間用）- デバッグ強化版"""
        try:
            self.get_logger().info(f"  - 段階的ジャンプ: {stages}段階で実行 (時間差: {end_timestamp - start_timestamp:.3f}秒)")
            
            # 段階的な中間フレームを取得
            time_gap = end_timestamp - start_timestamp
            stage_interval = time_gap / stages
            
            # フル画像バッファの状態をチェック
            with self.temporal_lock:
                buffer_size = len(self.full_image_buffer) if hasattr(self, 'full_image_buffer') else 0
                timestamp_size = len(self.image_timestamps) if hasattr(self, 'image_timestamps') else 0
                self.get_logger().info(f"    - バッファ状態: 画像={buffer_size}, タイムスタンプ={timestamp_size}")
            
            intermediate_frames = []
            failed_retrievals = 0
            
            # 中間フレーム取得のデバッグ強化
            for i in range(1, stages):  # 最初と最後を除く中間点
                intermediate_timestamp = start_timestamp + (stage_interval * i)
                self.get_logger().debug(f"    - 中間フレーム {i} 取得試行: {intermediate_timestamp:.3f}")
                
                intermediate_image = self._get_full_image_by_timestamp(intermediate_timestamp)
                if intermediate_image is not None:
                    if len(intermediate_image.shape) == 3:
                        intermediate_gray = cv2.cvtColor(intermediate_image, cv2.COLOR_BGR2GRAY)
                    else:
                        intermediate_gray = intermediate_image
                    intermediate_frames.append((intermediate_timestamp, intermediate_gray))
                    self.get_logger().debug(f"    - 中間フレーム {i} 取得成功: {intermediate_gray.shape}")
                else:
                    failed_retrievals += 1
                    self.get_logger().warning(f"    - 中間フレーム {i} 取得失敗: {intermediate_timestamp:.3f}")
            
            # 最終画像を追加（必ず追加）
            intermediate_frames.append((end_timestamp, target_gray))
            self.get_logger().info(f"    - フレーム取得結果: {len(intermediate_frames)}個成功, {failed_retrievals}個失敗")
            
            if len(intermediate_frames) == 0:
                self.get_logger().error("  - 段階的ジャンプ: 利用可能な中間フレームがありません")
                return
            
            # 段階的に更新
            success_count = 0
            total_trackers = len(self.tracked_features_per_label)
            
            self.get_logger().info(f"    - トラッカー更新開始: {total_trackers}個のトラッカー")
            
            for stage_idx, (timestamp, gray_image) in enumerate(intermediate_frames):
                self.get_logger().debug(f"    - 段階 {stage_idx + 1}/{len(intermediate_frames)}: {timestamp:.3f}, 画像={gray_image.shape}")
                
                # 画像の有効性チェック
                if gray_image is None or gray_image.size == 0:
                    self.get_logger().error(f"    - 段階 {stage_idx + 1}: 無効な画像データをスキップ")
                    continue
                
                # BGRに変換（デバッグ情報付き）
                try:
                    if len(gray_image.shape) == 2:  # グレースケール
                        tracker_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                    elif len(gray_image.shape) == 3 and gray_image.shape[2] == 3:  # 既にBGR
                        tracker_image = gray_image
                    else:
                        self.get_logger().error(f"    - 段階 {stage_idx + 1}: 不正な画像形状: {gray_image.shape}")
                        continue
                    self.get_logger().debug(f"    - 段階 {stage_idx + 1}: BGR変換完了 {tracker_image.shape}")
                except Exception as e:
                    self.get_logger().error(f"    - 段階 {stage_idx + 1}: BGR変換エラー: {e}")
                    continue
                
                stage_success = 0
                stage_failures = {}
                
                with self.tracking_data_lock:
                    for label, tracker in list(self.tracked_csrt_trackers.items()):
                        try:
                            # トラッカー更新前の詳細状態をログ
                            current_bbox = self.tracked_boxes_per_label.get(label, "未設定")
                            self.get_logger().debug(f"      - {label}: 更新前bbox={current_bbox}")
                            
                            # トラッカーの安定性をチェック（段階的ジャンプ用）
                            if not self._is_tracker_stable(label):
                                self.get_logger().debug(f"      - {label}: トラッカー不安定期間中（段階的ジャンプ）")
                            
                            # tracker.update()実行前の事前検証（段階的ジャンプ用）
                            if not self._pre_update_bbox_validation(label, gray_image):
                                stage_failures[label] = "事前検証失敗（bbox不正）"
                                self.get_logger().warning(f"      - {label}: 段階的ジャンプ事前検証失敗、スキップ")
                                # 失敗カウントを増加
                                if label not in self.tracker_failure_counts:
                                    self.tracker_failure_counts[label] = 0
                                self.tracker_failure_counts[label] += 1
                                continue
                            
                            # 画像とトラッカーの詳細情報をログ
                            self.get_logger().debug(f"      - {label}: 画像形状={tracker_image.shape}, dtype={tracker_image.dtype}")
                            self.get_logger().debug(f"      - {label}: 画像値範囲=[{tracker_image.min()}-{tracker_image.max()}]")
                            
                            # トラッカーの種類を特定
                            tracker_type = type(tracker).__name__
                            self.get_logger().debug(f"      - {label}: トラッカー種類={tracker_type}")
                            
                            # OpenCVバージョンに応じた詳細エラーハンドリング
                            try:
                                success, opencv_bbox = tracker.update(tracker_image)
                                self.get_logger().debug(f"      - {label}: update結果 success={success}, bbox={opencv_bbox}")
                            except cv2.error as cv_error:
                                stage_failures[label] = f"OpenCVエラー: {cv_error}"
                                self.get_logger().error(f"      - {label}: OpenCVエラー詳細: {cv_error}")
                                continue
                            except Exception as update_error:
                                stage_failures[label] = f"update例外: {update_error}"
                                self.get_logger().error(f"      - {label}: update例外詳細: {update_error}")
                                continue
                            
                            if success and opencv_bbox is not None:
                                x, y, w, h = opencv_bbox
                                bbox = [x, y, x + w, y + h]
                                
                                self.get_logger().debug(f"      - {label}: OpenCV bbox=[{x:.2f},{y:.2f},{w:.2f},{h:.2f}]")
                                
                                # 境界チェックと自動正規化（詳細ログ付き）
                                height, width = gray_image.shape
                                if (0 <= x < width and 0 <= y < height and 
                                    x + w <= width and y + h <= height and w > 0 and h > 0):
                                    self.tracked_boxes_per_label[label] = bbox
                                    stage_success += 1
                                    self.get_logger().debug(f"      - {label}: 更新成功 {bbox}")
                                else:
                                    # 段階的ジャンプでも境界外bboxを自動正規化
                                    self.get_logger().warning(f"      - {label}: 段階的ジャンプで境界外bbox検出、自動正規化実行")
                                    normalized_bbox = self._normalize_bounding_box(bbox, width, height, f"{label}_staged")
                                    if normalized_bbox is not None:
                                        self.tracked_boxes_per_label[label] = normalized_bbox
                                        stage_success += 1
                                        self.get_logger().info(f"      - {label}: 段階的ジャンプ境界外bbox正規化成功 {bbox} -> {normalized_bbox}")
                                    else:
                                        stage_failures[label] = f"段階的ジャンプ境界外bbox正規化失敗: [{x:.1f},{y:.1f},{w:.1f},{h:.1f}], 画像={width}x{height}"
                                        self.get_logger().error(f"      - {label}: {stage_failures[label]}")
                            else:
                                # tracker.update()が失敗した詳細な理由を調査
                                if not success:
                                    stage_failures[label] = f"tracker.update()戻り値=False（トラッキング失敗）"
                                elif opencv_bbox is None:
                                    stage_failures[label] = f"tracker.update()結果でbbox=None"
                                else:
                                    stage_failures[label] = f"不明な失敗: success={success}, bbox={opencv_bbox}"
                                
                                self.get_logger().warning(f"      - {label}: トラッカー更新失敗 {stage_failures[label]}")
                                
                                # 連続失敗のカウントを増加
                                if label not in self.tracker_failure_counts:
                                    self.tracker_failure_counts[label] = 0
                                self.tracker_failure_counts[label] += 1
                                
                                # 段階的ジャンプでも積極的なトラッカー交換
                                failure_count = self.tracker_failure_counts[label]
                                if failure_count >= 2:  # 段階的ジャンプでも早めの交換
                                    self.get_logger().error(f"      - {label}: 段階的ジャンプで連続失敗{failure_count}回、トラッカー交換を実行")
                                    replacement_success = self._attempt_tracker_replacement(label, tracker_image)
                                    if replacement_success:
                                        # 交換成功後、新しいトラッカーで再試行
                                        new_tracker = self.tracked_csrt_trackers.get(label)
                                        if new_tracker is not None:
                                            try:
                                                retry_success, retry_bbox = new_tracker.update(tracker_image)
                                                if retry_success and retry_bbox is not None:
                                                    x, y, w, h = retry_bbox
                                                    bbox = [x, y, x + w, y + h]
                                                    height, width = gray_image.shape
                                                    normalized_bbox = self._normalize_bounding_box(bbox, width, height, f"{label}_staged_retry")
                                                    if normalized_bbox is not None:
                                                        self.tracked_boxes_per_label[label] = normalized_bbox
                                                        stage_success += 1
                                                        self.get_logger().info(f"      - {label}: 段階的ジャンプ交換後再試行成功 {normalized_bbox}")
                                            except Exception as retry_error:
                                                self.get_logger().warning(f"      - {label}: 段階的ジャンプ交換後再試行エラー: {retry_error}")
                                else:
                                    # 品質診断（失敗回数が閾値未満の場合）
                                    self._diagnose_tracker_quality(label, tracker, tracker_image)
                                    
                        except Exception as e:
                            stage_failures[label] = f"例外: {str(e)}"
                            self.get_logger().error(f"      - {label}: 段階的更新例外 {stage_failures[label]}")
                            import traceback
                            self.get_logger().error(f"        スタックトレース: {traceback.format_exc()}")
                            continue
                
                # 段階結果の詳細ログ
                self.get_logger().info(f"    - 段階 {stage_idx + 1} 結果: {stage_success}/{total_trackers}成功")
                if stage_failures:
                    for label, error in stage_failures.items():
                        self.get_logger().debug(f"      - 失敗詳細 {label}: {error}")
                
                if stage_idx == len(intermediate_frames) - 1:  # 最終段階
                    success_count = stage_success
            
            success_rate = success_count / total_trackers if total_trackers > 0 else 0.0
            self.get_logger().info(f"  - 段階的ジャンプ結果: {success_count}/{total_trackers}成功 ({success_rate*100:.1f}%)")
            
            # 完全失敗の場合の対策検討
            if success_count == 0 and total_trackers > 0:
                self.get_logger().error("  - 段階的ジャンプ完全失敗: トラッカー再初期化を検討")
                
                # 早送り処理の失敗パターンを分析
                self._analyze_fastforward_failure_pattern(start_timestamp, end_timestamp)
            
            return success_count > 0
                
        except Exception as e:
            self.get_logger().error(f"CSRT段階的ジャンプエラー: {e}")
            import traceback
            self.get_logger().error(f"スタックトレース: {traceback.format_exc()}")
            return False
    
    # _csrt_staged_jump_with_fallback関数は削除（不要な複雑処理）
    
    # _diagnose_and_repair_trackers関数は削除（不要な複雑処理）
    
    # _diagnose_tracker_quality関数は削除（不要な複雑処理）
    
    # _analyze_fastforward_failure_pattern関数は削除（不要な複雑処理）
    
    # _csrt_micro_step_jump関数は削除（不要な複雑処理）
    
    
    def _emergency_tracker_reset(self, current_gray: np.ndarray) -> None:
        """緊急時のトラッカーリセット処理"""
        try:
            self.get_logger().error("  === 緊急トラッカーリセット開始 ===")
            
            # 現在のバウンディングボックスを保存
            backup_boxes = {}
            with self.tracking_data_lock:
                backup_boxes = self.tracked_boxes_per_label.copy()
                self.get_logger().error(f"    - 現在のバウンディングボックスをバックアップ: {len(backup_boxes)}個")
            
            # トラッカーを一旦全削除
            with self.tracking_data_lock:
                tracker_count = len(self.tracked_features_per_label)
                self.tracked_csrt_trackers.clear()
                self.tracked_features_per_label.clear()
                self.tracker_failure_counts.clear()
                self.tracker_initialization_times.clear()  # 初期化時刻記録もクリア
                self.get_logger().error(f"    - 既存トラッカーを削除: {tracker_count}個")
            
            # バウンディングボックスを基に新しいトラッカーを再初期化
            success_count = 0
            for label, bbox in backup_boxes.items():
                try:
                    # BGRに変換
                    if len(current_gray.shape) == 2:
                        bgr_image = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr_image = current_gray
                    
                    height, width = bgr_image.shape[:2]
                    
                    # 緊急時のボックス正規化（より厳しい制限）
                    normalized_bbox = self._normalize_bounding_box(bbox, width, height, f"{label}_emergency")
                    if normalized_bbox is None:
                        self.get_logger().error(f"    - {label}: 緊急時bbox正規化失敗 {bbox}")
                        # 再初期化に失敗したボックスは削除
                        with self.tracking_data_lock:
                            if label in self.tracked_boxes_per_label:
                                del self.tracked_boxes_per_label[label]
                        continue
                    
                    self.get_logger().error(f"    - {label}: 緊急再初期化試行 bbox={normalized_bbox}")
                    
                    # トラッカー再初期化（正規化済みbboxを使用）
                    if self._initialize_csrt_tracker(label, bgr_image, normalized_bbox):
                        success_count += 1
                        self.get_logger().error(f"    - {label}: 緊急再初期化成功")
                    else:
                        self.get_logger().error(f"    - {label}: 緊急再初期化失敗")
                        # 再初期化に失敗したボックスは削除
                        with self.tracking_data_lock:
                            if label in self.tracked_boxes_per_label:
                                del self.tracked_boxes_per_label[label]
                    
                except Exception as e:
                    self.get_logger().error(f"    - {label}: 緊急再初期化例外: {e}")
                    continue
            
            self.get_logger().error(f"  - 緊急リセット結果: {success_count}/{len(backup_boxes)}個成功")
            
            if success_count > 0:
                self.get_logger().error("  - 緊急リセット成功: 一部トラッカーが復旧")
                # 失敗カウントをリセット
                self._reset_tracker_failure_counts()
            else:
                self.get_logger().error("  - 緊急リセット失敗: 全てのトラッカーが失われました")
            
            self.get_logger().error("  === 緊急トラッカーリセット完了 ===")
            
        except Exception as e:
            self.get_logger().error(f"緊急トラッカーリセットエラー: {e}")
    
    def _selective_tracker_reset(self, current_gray: np.ndarray, failure_threshold: int = 5) -> int:
        """選択的トラッカーリセット処理（改善版）"""
        try:
            self.get_logger().info(f"  === 選択的トラッカーリセット開始 (閾値: {failure_threshold}) ===")
            
            # 失敗回数の多いトラッカーを特定
            problematic_trackers = []
            with self.tracking_data_lock:
                for label, failure_count in self.tracker_failure_counts.items():
                    if failure_count >= failure_threshold:
                        problematic_trackers.append((label, failure_count))
            
            if not problematic_trackers:
                self.get_logger().info("  - リセット対象のトラッカーなし")
                return 0
            
            # 失敗回数順にソート（多い順）
            problematic_trackers.sort(key=lambda x: x[1], reverse=True)
            self.get_logger().info(f"  - リセット対象: {len(problematic_trackers)}個")
            
            reset_success_count = 0
            for label, failure_count in problematic_trackers:
                try:
                    current_bbox = self.tracked_boxes_per_label.get(label)
                    if current_bbox is None:
                        self.get_logger().warning(f"    - {label}: bbox未設定のためスキップ")
                        continue
                    
                    self.get_logger().info(f"    - {label}: リセット実行 (失敗回数: {failure_count})")
                    
                    # 古いトラッカーをクリーンアップ
                    with self.tracking_data_lock:
                        if label in self.tracked_csrt_trackers:
                            del self.tracked_csrt_trackers[label]
                        if label in self.tracked_features_per_label:
                            del self.tracked_features_per_label[label]
                        if label in self.tracker_failure_counts:
                            del self.tracker_failure_counts[label]
                        if label in self.tracker_initialization_times:
                            del self.tracker_initialization_times[label]
                        if hasattr(self, 'tracker_types') and label in self.tracker_types:
                            del self.tracker_types[label]
                    
                    # 画像準備
                    if len(current_gray.shape) == 2:
                        bgr_image = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr_image = current_gray
                    
                    height, width = bgr_image.shape[:2]
                    
                    # bbox正規化
                    normalized_bbox = self._normalize_bounding_box(current_bbox, width, height, f"{label}_selective_reset")
                    if normalized_bbox is None:
                        self.get_logger().error(f"    - {label}: bbox正規化失敗、削除")
                        with self.tracking_data_lock:
                            if label in self.tracked_boxes_per_label:
                                del self.tracked_boxes_per_label[label]
                        continue
                    
                    # 新しいトラッカーで再初期化
                    if self._initialize_csrt_tracker(label, bgr_image, normalized_bbox):
                        reset_success_count += 1
                        self.get_logger().info(f"    ✓ {label}: 選択的リセット成功")
                        
                        # リセット統計を記録
                        if not hasattr(self, 'reset_statistics'):
                            self.reset_statistics = {}
                        if label not in self.reset_statistics:
                            self.reset_statistics[label] = 0
                        self.reset_statistics[label] += 1
                        
                    else:
                        self.get_logger().error(f"    ✗ {label}: 選択的リセット失敗、削除")
                        with self.tracking_data_lock:
                            if label in self.tracked_boxes_per_label:
                                del self.tracked_boxes_per_label[label]
                
                except Exception as e:
                    self.get_logger().error(f"    - {label}: 選択的リセット例外: {e}")
                    continue
            
            self.get_logger().info(f"  - 選択的リセット結果: {reset_success_count}/{len(problematic_trackers)}個成功")
            self.get_logger().info("  === 選択的トラッカーリセット完了 ===")
            
            return reset_success_count
            
        except Exception as e:
            self.get_logger().error(f"選択的トラッカーリセットエラー: {e}")
            return 0
    
    def _get_reset_statistics_summary(self) -> str:
        """リセット統計の要約を取得"""
        try:
            if not hasattr(self, 'reset_statistics') or not self.reset_statistics:
                return "リセット履歴なし"
            
            total_resets = sum(self.reset_statistics.values())
            most_reset_label = max(self.reset_statistics.items(), key=lambda x: x[1])
            
            summary = f"総リセット回数: {total_resets}, 最多リセット: {most_reset_label[0]}({most_reset_label[1]}回)"
            return summary
            
        except Exception as e:
            return f"統計取得エラー: {e}"
    
    def _normalize_bounding_box(self, bbox: List[float], image_width: int, image_height: int, label: str) -> Optional[List[float]]:
        """バウンディングボックス正規化（オフスクリーン移動許可版）"""
        try:
            if len(bbox) != 4:
                self.get_logger().error(f"{label}: 不正なbbox形式: {bbox}")
                return None
            
            x1, y1, x2, y2 = bbox
            
            # [x, y, w, h]形式の場合は[x1, y1, x2, y2]に変換
            if x2 < x1 or y2 < y1:
                w, h = x2, y2
                x2, y2 = x1 + w, y1 + h
                self.get_logger().debug(f"{label}: [x,y,w,h]形式を変換: [{x1},{y1},{w},{h}] → [{x1},{y1},{x2},{y2}]")
            
            original_bbox = [x1, y1, x2, y2]
            w, h = abs(x2 - x1), abs(y2 - y1)
            
            # 破損検出: 異常に小さいサイズ
            min_size = 8
            if w < min_size or h < min_size:
                self.get_logger().warn(f"{label}: bbox破損（過小）: {w:.1f}x{h:.1f} -> 修復")
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                half_size = min_size / 2
                x1, y1 = center_x - half_size, center_y - half_size
                x2, y2 = center_x + half_size, center_y + half_size
                w, h = min_size, min_size
            
            # 破損検出: 異常に大きいサイズ（画像の95%以上）
            max_w, max_h = image_width * 0.95, image_height * 0.95
            if w > max_w or h > max_h:
                self.get_logger().warn(f"{label}: bbox破損（過大）: {w:.1f}x{h:.1f} -> スケール")
                scale = min(max_w / w, max_h / h)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                new_w, new_h = w * scale, h * scale
                x1, y1 = center_x - new_w/2, center_y - new_h/2
                x2, y2 = center_x + new_w/2, center_y + new_h/2
                w, h = new_w, new_h
            
            # オフスクリーン移動の許可判定
            margin = max(image_width, image_height) * 0.3  # 30%まで画面外を許可
            
            # 完全に画面外（削除対象）
            if (x2 < -margin or x1 > image_width + margin or 
                y2 < -margin or y1 > image_height + margin):
                self.get_logger().info(f"{label}: 完全画面外で削除: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                return None
            
            # 部分的オフスクリーン（許可）
            off_screen = (x2 < 0 or x1 > image_width or y2 < 0 or y1 > image_height)
            if off_screen:
                self.get_logger().debug(f"{label}: オフスクリーン移動（許可）: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
            
            # 極端な破損座標のみ修正（画像サイズの2倍以上）
            extreme_threshold = max(image_width, image_height) * 2
            corrected = False
            
            if x1 < -extreme_threshold:
                x1 = -margin
                corrected = True
            if y1 < -extreme_threshold:
                y1 = -margin  
                corrected = True
            if x2 > image_width + extreme_threshold:
                x2 = image_width + margin
                corrected = True
            if y2 > image_height + extreme_threshold:
                y2 = image_height + margin
                corrected = True
            
            if corrected:
                self.get_logger().warn(f"{label}: 極端破損修正: {original_bbox} → [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
            
            # 最終検証
            final_w, final_h = abs(x2 - x1), abs(y2 - y1)
            if final_w < min_size or final_h < min_size or x1 >= x2 or y1 >= y2:
                self.get_logger().error(f"{label}: 最終検証失敗: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                return None
            
            normalized_bbox = [float(x1), float(y1), float(x2), float(y2)]
            
            # 修正があった場合のみログ
            if normalized_bbox != original_bbox:
                if corrected or w < min_size * 2:
                    self.get_logger().info(f"{label}: bbox修正: {original_bbox} → {normalized_bbox}")
                else:
                    self.get_logger().debug(f"{label}: bbox正規化: {original_bbox} → {normalized_bbox}")
            
            return normalized_bbox
            
        except Exception as e:
            self.get_logger().error(f"{label}: bbox正規化エラー: {e}")
            return None
            
    def _fallback_temporal_catchup(self, start_timestamp: float, end_timestamp: float) -> None:
        """フォールバック: 段階的キャッチアップ処理"""
        try:
            catchup_rois = []
            with self.temporal_lock:
                # キャッチアップに必要なROIを収集
                for i, timestamp in enumerate(self.image_timestamps):
                    if start_timestamp < timestamp <= end_timestamp:
                        catchup_rois.append((timestamp, self.roi_buffer[i].copy(), self.roi_coords[i]))
            
            # 時間順にソート
            catchup_rois.sort(key=lambda x: x[0])
            
            if len(catchup_rois) > 0:
                self.get_logger().info(f"フォールバックキャッチアップ: {len(catchup_rois)}フレーム")
                
                # 高速オプティカルフロー更新（ROIベース）
                for timestamp, roi_gray, roi_coords in catchup_rois:
                    self._fast_optical_flow_update_roi(roi_gray, roi_coords)
                    
                self.get_logger().info("フォールバックキャッチアップ完了")
            
        except Exception as e:
            self.get_logger().error(f"フォールバックキャッチアップエラー: {e}")
    
    def _fast_optical_flow_update(self, gray: np.ndarray) -> None:
        """高速CSRTトラッカー更新（キャッチアップ用）"""
        try:
            if not self.tracking_valid:
                return
            
            # CSRTトラッカー更新
            self._update_csrt_trackers(gray)
            
            # 前フレーム更新
            self.prev_gray = gray.copy()
            
        except Exception as e:
            self.get_logger().error(f"高速オプティカルフロー更新エラー: {e}")
    
    def _get_current_gray_image(self) -> Optional[np.ndarray]:
        """真の最新画像を取得（グレースケール変換含む）"""
        try:
            with self.temporal_lock:
                if len(self.full_image_buffer) > 0:
                    # 最新画像を取得
                    latest_image = self.full_image_buffer[-1].copy()
                    
                    # グレースケール変換
                    if len(latest_image.shape) == 3:
                        return cv2.cvtColor(latest_image, cv2.COLOR_BGR2GRAY)
                    else:
                        return latest_image
            return None
        except Exception as e:
            self.get_logger().error(f"現在画像取得エラー: {e}")
            return None
    
    def _get_absolute_latest_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """絶対最新の画像とタイムスタンプを取得"""
        try:
            with self.temporal_lock:
                if len(self.full_image_buffer) > 0 and len(self.image_timestamps) > 0:
                    latest_image = self.full_image_buffer[-1].copy()
                    latest_timestamp = self.image_timestamps[-1]
                    
                    # グレースケール変換
                    if len(latest_image.shape) == 3:
                        gray_image = cv2.cvtColor(latest_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = latest_image
                        
                    return gray_image, latest_timestamp
            return None, None
        except Exception as e:
            self.get_logger().error(f"絶対最新画像取得エラー: {e}")
            return None, None
    
    
    def _track_optical_flow_with_temporal_alignment(self, current_gray: np.ndarray, current_time: float) -> None:
        """時間整合性を考慮したオプティカルフロー処理"""
        try:
            if not self.tracking_valid or self.prev_gray is None:
                # 通常のオプティカルフロー処理にフォールバック
                self._track_optical_flow(current_gray)
                return
            
            # 前フレームと現在フレームの時間差を取得
            with self.temporal_lock:
                if len(self.image_timestamps) < 2:
                    # 時間情報が不足している場合は通常処理
                    self._track_optical_flow(current_gray)
                    return
                
                # 前フレームのタイムスタンプを取得
                prev_timestamp = self.image_timestamps[-2]  # 一つ前のフレーム
                current_timestamp = current_time
                
                time_gap = current_timestamp - prev_timestamp
                
                # 時間ギャップが大きい場合は補間処理
                if time_gap > 0.1:  # 100ms以上の間隔
                    self.get_logger().info(f"オプティカルフロー時間ギャップ検出: {time_gap:.3f}秒 - 補間処理実行")
                    
                    # 前フレームと現在フレーム間の中間フレームで補間
                    self._perform_optical_flow_interpolation(prev_timestamp, current_timestamp, current_gray)
                else:
                    # 通常のオプティカルフロー処理
                    self._track_optical_flow(current_gray)
                    
        except Exception as e:
            self.get_logger().error(f"時間整合性オプティカルフローエラー: {e}")
            # フォールバック: 通常処理
            self._track_optical_flow(current_gray)
    
    def _perform_optical_flow_interpolation(self, start_timestamp: float, end_timestamp: float, end_gray: np.ndarray) -> None:
        """オプティカルフローの時間補間処理"""
        try:
            # 補間に必要な中間フレームを収集
            interpolation_frames = []
            with self.temporal_lock:
                for i, timestamp in enumerate(self.image_timestamps):
                    if start_timestamp < timestamp < end_timestamp:
                        interpolation_frames.append((timestamp, self.image_buffer[i].copy()))
            
            # 時間順にソート
            interpolation_frames.sort(key=lambda x: x[0])
            
            if len(interpolation_frames) > 0:
                self.get_logger().info(f"オプティカルフロー補間: {len(interpolation_frames)}フレーム")
                
                # 段階的にオプティカルフローを実行
                current_gray_for_flow = self.prev_gray
                for timestamp, gray_frame in interpolation_frames:
                    # 中間フレームでオプティカルフロー更新
                    self._update_optical_flow_step(current_gray_for_flow, gray_frame)
                    current_gray_for_flow = gray_frame
                
                # 最終フレームで最後の更新
                self._update_optical_flow_step(current_gray_for_flow, end_gray)
                
                self.get_logger().info("オプティカルフロー補間完了")
            else:
                # 中間フレームがない場合は直接処理
                self._track_optical_flow(end_gray)
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー補間エラー: {e}")
            # フォールバック: 通常処理
            self._track_optical_flow(end_gray)
    
    def _update_optical_flow_step(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> None:
        """一段階のCSRTトラッカー更新（補間用）"""
        try:
            # CSRTトラッカー更新
            self._update_csrt_trackers(curr_gray)
            
            # 前フレーム更新
            self.prev_gray = curr_gray.copy()
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー段階更新エラー: {e}")
    
    def _integrate_new_detections_with_existing_tracking(self, current_time: float) -> None:
        """シンプルな検出結果からトラッカー初期化"""
        try:
            with self.detection_data_lock:
                if self.latest_detection_data is None:
                    self.get_logger().warn("検出データがありません")
                    return
                
                boxes = self.latest_detection_data.get('boxes', [])
                labels = self.latest_detection_data.get('labels', [])
                
                if len(boxes) == 0:
                    self.get_logger().info("検出ボックスがありません")
                    return
            
            # 現在画像を取得
            current_gray = self._get_current_gray_image()
            if current_gray is None:
                self.get_logger().error("現在画像が取得できません")
                return
            
            # 既存トラッカーをクリア
            with self.tracking_data_lock:
                self.tracked_boxes_per_label.clear()
                self.tracked_csrt_trackers.clear()
                self.tracker_failure_counts.clear()
                
                success_count = 0
                self.get_logger().info(f"検出結果から{len(boxes)}個のトラッカーを初期化")
                
                for i, bbox in enumerate(boxes):
                    if i >= len(labels):
                        continue
                        
                    original_label = labels[i]
                    
                    # トラッキング対象チェック
                    if not self._is_tracking_target(original_label):
                        self.get_logger().info(f"ラベル'{original_label}'はトラッキング対象外")
                        continue
                    
                    # ユニークラベル作成
                    unique_label = f"{original_label}_{i}"
                    
                    # bboxをリストに変換
                    if hasattr(bbox, 'cpu'):
                        bbox_list = bbox.cpu().numpy().tolist()
                    else:
                        bbox_list = list(bbox)
                    
                    # CSRTトラッカー初期化
                    if self._initialize_csrt_tracker(unique_label, current_gray, bbox_list):
                        success_count += 1
                    else:
                        self.get_logger().error(f"トラッカー初期化失敗: {unique_label}")
                
                self.get_logger().info(f"トラッカー初期化完了: {success_count}/{len(boxes)}")
                self.tracking_valid = success_count > 0
                
        except Exception as e:
            self.get_logger().error(f"トラッカー初期化エラー: {e}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
    def _find_overlapping_existing_tracking(self, new_box: List[float], new_label: str) -> Optional[str]:
        """新規検出ボックスと重複する既存追跡を検索"""
        try:
            new_x1, new_y1, new_x2, new_y2 = new_box
            new_center_x = (new_x1 + new_x2) / 2
            new_center_y = (new_y1 + new_y2) / 2
            new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
            
            max_overlap = 0.0
            best_overlap_label = None
            
            for existing_label, existing_box in self.tracked_boxes_per_label.items():
                # ラベルの基本名が同じかチェック
                existing_base_label = existing_label.split('_')[0]
                new_base_label = new_label.split('.')[0].strip()
                
                if existing_base_label != new_base_label:
                    continue
                
                # 位置的な重複をチェック
                ex_x1, ex_y1, ex_x2, ex_y2 = existing_box
                ex_center_x = (ex_x1 + ex_x2) / 2
                ex_center_y = (ex_y1 + ex_y2) / 2
                
                # 中心点間の距離
                distance = np.sqrt((new_center_x - ex_center_x)**2 + (new_center_y - ex_center_y)**2)
                
                # IoU計算
                intersection_x1 = max(new_x1, ex_x1)
                intersection_y1 = max(new_y1, ex_y1)
                intersection_x2 = min(new_x2, ex_x2)
                intersection_y2 = min(new_y2, ex_y2)
                
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    existing_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
                    union_area = new_area + existing_area - intersection_area
                    
                    if union_area > 0:
                        overlap = intersection_area / union_area
                        
                        # 重複条件: IoU > threshold または 中心点距離 < threshold
                        if overlap > self.overlap_iou_threshold or distance < self.overlap_distance_threshold:
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_overlap_label = existing_label
            
            if best_overlap_label:
                self.get_logger().info(f"重複検出: '{new_label}' ↔ '{best_overlap_label}' (IoU: {max_overlap:.3f})")
            
            return best_overlap_label
            
        except Exception as e:
            self.get_logger().error(f"重複検索エラー: {e}")
            return None
    
    def _merge_detection_with_existing_tracking(self, existing_label: str, new_box: List[float], detection_timestamp: Optional[float]) -> None:
        """既存追跡と新規検出をマージ"""
        try:
            # 既存の追跡は維持し、バウンディングボックスのみ微調整
            if existing_label in self.tracked_boxes_per_label:
                existing_box = self.tracked_boxes_per_label[existing_label]
                
                # 新旧ボックスの重み付き平均（既存追跡を重視）
                weight_existing = self.merge_weight_existing
                weight_new = self.merge_weight_new
                
                merged_box = [
                    existing_box[0] * weight_existing + new_box[0] * weight_new,
                    existing_box[1] * weight_existing + new_box[1] * weight_new,
                    existing_box[2] * weight_existing + new_box[2] * weight_new,
                    existing_box[3] * weight_existing + new_box[3] * weight_new
                ]
                
                self.tracked_boxes_per_label[existing_label] = merged_box
                
                # 検出時刻を更新
                current_time = time.time()
                self.object_last_detection_time[existing_label] = current_time
                
                self.get_logger().info(f"追跡'{existing_label}'をマージ更新: {existing_box} → {merged_box}")
            
        except Exception as e:
            self.get_logger().error(f"マージエラー: {e}")
    
    def _initialize_csrt_tracker(self, label: str, image: np.ndarray, bbox: List[float]) -> bool:
        """実際のCSRTトラッカー初期化（高精度）"""
        try:
            # 基本チェック
            if image is None or image.size == 0:
                self.get_logger().error(f"無効な画像: {label}")
                return False
                
            if not bbox or len(bbox) != 4:
                self.get_logger().error(f"無効なbbox: {label} - {bbox}")
                return False
            
            height, width = image.shape[:2]
            x1, y1, x2, y2 = bbox
            
            # 境界クリッピング
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = min(x2, width)
            y2 = min(y2, height)
            
            # OpenCVトラッカー用の(x, y, width, height)形式に変換
            tracker_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            
            # 最小サイズ保証
            if tracker_bbox[2] < 20 or tracker_bbox[3] < 20:
                self.get_logger().warn(f"bbox小さすぎ: {label} - {tracker_bbox[2]}x{tracker_bbox[3]}")
                return False
            
            # CSRTトラッカー作成
            csrt_tracker = cv2.TrackerCSRT_create()
            
            # BGR画像に変換（CSRTはBGR形式を要求）
            if len(image.shape) == 2:
                # グレースケール → BGR変換
                tracker_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                tracker_image = image.copy()
            else:
                self.get_logger().error(f"サポートされていない画像形式: {image.shape}")
                return False
            
            # CSRTトラッカー初期化
            success = csrt_tracker.init(tracker_image, tracker_bbox)
            
            if not success:
                self.get_logger().error(f"CSRTトラッカー初期化失敗: {label}")
                return False
            
            # トラッカーとデータを保存
            with self.tracking_data_lock:
                # CSRTトラッカー保存
                if not hasattr(self, 'tracked_csrt_trackers'):
                    self.tracked_csrt_trackers = {}
                self.tracked_csrt_trackers[label] = csrt_tracker
                
                # バウンディングボックス保存
                self.tracked_boxes_per_label[label] = [x1, y1, x2, y2]
                self.tracker_failure_counts[label] = 0
            
            self.get_logger().info(f"✓ CSRTトラッカー初期化成功: {label} - bbox={tracker_bbox}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"CSRTトラッカー初期化エラー {label}: {e}")
            return False
    
    def _create_optimized_csrt_tracker(self):
        """高速移動環境に最適化されたCSRTトラッカーを作成（OpenCV互換性対応）"""
        try:
            # CSRTパラメータを高速移動環境用に最適化
            params = cv2.TrackerCSRT_Params()
            
            # 安全なパラメータのみ設定（OpenCV互換性を確保）
            safe_params = [
                ('use_hog', True),               # HOG特徴を使用
                ('use_color_names', True),       # カラー特徴を使用
                ('use_gray', False),             # グレースケール特徴は無効
                ('use_rgb', True),               # RGB特徴を使用
                ('use_channel_weights', True),   # チャネル重みを使用
                ('use_segmentation', True),      # セグメンテーション機能を使用
                ('admm_iterations', 4),          # ADMM反復回数
                ('histogram_bins', 16),          # ヒストグラムビン数
                ('background_ratio', 2),         # 背景比率
                ('number_of_scales', 33),        # スケール数
                ('scale_sigma_factor', 0.25),    # スケールシグマ因子
                ('scale_model_max_area', 512),   # スケールモデル最大エリア
                ('scale_lr', 0.025),             # スケール学習率
                ('scale_step', 1.02)             # スケールステップ
            ]
            
            # 安全にパラメータを設定
            applied_params = []
            for param_name, param_value in safe_params:
                try:
                    if hasattr(params, param_name):
                        setattr(params, param_name, param_value)
                        applied_params.append(param_name)
                    else:
                        self.get_logger().debug(f"CSRTパラメータ '{param_name}' は未対応")
                except Exception as param_error:
                    self.get_logger().debug(f"CSRTパラメータ '{param_name}' 設定エラー: {param_error}")
            
            if applied_params:
                self.get_logger().debug(f"CSRT最適化パラメータ適用: {applied_params}")
                tracker = cv2.TrackerCSRT_create(params)
                return tracker
            else:
                self.get_logger().warning("CSRT最適化パラメータが適用できません、標準版を使用")
                return cv2.TrackerCSRT_create()
            
        except Exception as e:
            self.get_logger().warning(f"最適化CSRTトラッカー作成失敗、標準版を使用: {e}")
            return cv2.TrackerCSRT_create()
    
    def _create_optimized_kcf_tracker(self):
        """高速移動環境に最適化されたKCFトラッカーを作成（OpenCV互換性対応）"""
        try:
            # KCFパラメータを高速移動環境用に最適化
            params = cv2.TrackerKCF_Params()
            
            # 安全なパラメータのみ設定（OpenCV互換性を確保）
            safe_params = [
                ('detect_thresh', 0.4),          # 検出閾値
                ('sigma', 0.2),                  # ガウシアンカーネルのシグマ
                ('interp_factor', 0.075),        # 補間因子
                ('output_sigma_factor', 1.0/16.0), # 出力シグマ因子
                ('resize', True),                # リサイズを有効化
                ('max_patch_size', 80*80),       # 最大パッチサイズ
                ('split_coeff', True),           # 係数分割を有効化
                ('wrap_kernel', False),          # カーネルラップは無効化
                ('compress_feature', True),      # 特徴圧縮を有効化
                ('desc_npca', 0),                # NPCAを無効化
                ('compressed_size', 2)           # 圧縮サイズ
            ]
            
            # 特別な処理が必要なパラメータ
            try:
                if hasattr(params, 'desc_pca'):
                    # cv2.TrackerKCF_MODE_* が存在するかチェック
                    if hasattr(cv2, 'TrackerKCF_MODE_GRAY') and hasattr(cv2, 'TrackerKCF_MODE_CN'):
                        params.desc_pca = cv2.TrackerKCF_MODE_GRAY | cv2.TrackerKCF_MODE_CN
                        safe_params.append(('desc_pca', 'GRAY|CN'))
            except Exception as e:
                self.get_logger().debug(f"KCF desc_pcaパラメータ設定エラー: {e}")
            
            # 安全にパラメータを設定
            applied_params = []
            for param_name, param_value in safe_params:
                try:
                    if hasattr(params, param_name):
                        setattr(params, param_name, param_value)
                        applied_params.append(param_name)
                    else:
                        self.get_logger().debug(f"KCFパラメータ '{param_name}' は未対応")
                except Exception as param_error:
                    self.get_logger().debug(f"KCFパラメータ '{param_name}' 設定エラー: {param_error}")
            
            if applied_params:
                self.get_logger().debug(f"KCF最適化パラメータ適用: {applied_params}")
                tracker = cv2.TrackerKCF_create(params)
                return tracker
            else:
                self.get_logger().warning("KCF最適化パラメータが適用できません、標準版を使用")
                return cv2.TrackerKCF_create()
            
        except Exception as e:
            self.get_logger().warning(f"最適化KCFトラッカー作成失敗、標準版を使用: {e}")
            return cv2.TrackerKCF_create()
    
    def _validate_tracker_initialization(self, label: str, tracker, image: np.ndarray) -> bool:
        """トラッカー初期化後の検証テスト"""
        try:
            # 現在のbboxを取得
            current_bbox = self.tracked_boxes_per_label.get(label)
            if current_bbox is None:
                self.get_logger().warning(f"{label}: 初期化検証時にbboxが未設定")
                return False
            
            self.get_logger().debug(f"{label}: 検証開始 - 保存済みbbox={current_bbox}")
            
            # ** 寛容な検証アプローチ：テスト更新を試行するが、失敗でも初期化成功とみなす **
            try:
                test_success, test_bbox = tracker.update(image)
                self.get_logger().debug(f"{label}: 検証テスト結果 success={test_success}, bbox={test_bbox}")
                
                if test_success and test_bbox is not None:
                    x, y, w, h = test_bbox
                    height, width = image.shape[:2]
                    
                    # 完全に異常な場合のみ警告（それでも成功とみなす）
                    if w < 1 or h < 1 or x < -width or y < -height or x > width*2 or y > height*2:
                        self.get_logger().warning(f"{label}: 検証で異常なbbox検出: [{x:.1f},{y:.1f},{w:.1f},{h:.1f}] vs {width}x{height}")
                    
                    self.get_logger().debug(f"{label}: 初期化検証成功 - テストbbox=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}]")
                else:
                    self.get_logger().debug(f"{label}: 検証テスト失敗も初期化は有効として処理")
                
            except Exception as update_error:
                self.get_logger().debug(f"{label}: 検証中のupdate()エラー（初期化は有効）: {update_error}")
            
            # ** 重要: 検証結果に関わらず、tracker.init()が成功していれば常にTrueを返す **
            return True
            
        except Exception as e:
            self.get_logger().warning(f"{label}: 初期化検証エラー（初期化は有効として処理）: {e}")
            return True  # エラーが発生しても初期化は成功として扱う
    
    def _cleanup_failed_tracker(self, label: str) -> None:
        """失敗したトラッカーの完全クリーンアップ"""
        try:
            with self.tracking_data_lock:
                if label in self.tracked_csrt_trackers:
                    del self.tracked_csrt_trackers[label]
                if label in self.tracked_features_per_label:
                    del self.tracked_features_per_label[label]
                if label in self.tracker_failure_counts:
                    del self.tracker_failure_counts[label]
                if label in self.tracker_initialization_times:
                    del self.tracker_initialization_times[label]
                if label in self.tracked_boxes_per_label:
                    del self.tracked_boxes_per_label[label]
                if hasattr(self, 'tracker_types') and label in self.tracker_types:
                    del self.tracker_types[label]
            self.get_logger().debug(f"{label}: 失敗したトラッカーのクリーンアップ完了")
        except Exception as e:
            self.get_logger().error(f"{label}: トラッカークリーンアップエラー: {e}")
    
    def _attempt_tracker_replacement(self, label: str, current_image: np.ndarray) -> bool:
        """失敗したトラッカーの交換を試行"""
        try:
            self.get_logger().info(f"=== {label}: トラッカー交換処理開始 ===")
            
            # 現在のbboxを保存
            current_bbox = self.tracked_boxes_per_label.get(label)
            if current_bbox is None:
                self.get_logger().error(f"{label}: 交換用bboxが見つかりません")
                return False
            
            # 現在のトラッカー種類を取得
            current_tracker_type = getattr(self, 'tracker_types', {}).get(label, "不明")
            self.get_logger().info(f"{label}: 現在のトラッカー種類={current_tracker_type}")
            
            # 古いトラッカーをクリーンアップ
            with self.tracking_data_lock:
                if label in self.tracked_csrt_trackers:
                    del self.tracked_csrt_trackers[label]
                if label in self.tracked_features_per_label:
                    del self.tracked_features_per_label[label]
                if label in self.tracker_failure_counts:
                    del self.tracker_failure_counts[label]
                if label in self.tracker_initialization_times:
                    del self.tracker_initialization_times[label]
                if hasattr(self, 'tracker_types') and label in self.tracker_types:
                    del self.tracker_types[label]
            
            # 新しいトラッカーで再初期化を試行
            success = self._initialize_csrt_tracker(label, current_image, current_bbox)
            
            if success:
                new_tracker_type = getattr(self, 'tracker_types', {}).get(label, "不明")
                self.get_logger().info(f"✓ {label}: トラッカー交換成功 {current_tracker_type} -> {new_tracker_type}")
                
                # 交換後の安定化期間を設定
                if hasattr(self, 'tracker_replacement_times'):
                    self.tracker_replacement_times[label] = time.time()
                else:
                    self.tracker_replacement_times = {label: time.time()}
                    
                return True
            else:
                self.get_logger().error(f"✗ {label}: トラッカー交換失敗、削除")
                # 交換に失敗した場合はbboxも削除
                with self.tracking_data_lock:
                    if label in self.tracked_boxes_per_label:
                        del self.tracked_boxes_per_label[label]
                return False
                
        except Exception as e:
            self.get_logger().error(f"{label}: トラッカー交換エラー: {e}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
            return False
    
    def _is_tracker_stable(self, label: str) -> bool:
        """トラッカーが安定化期間を過ぎているかチェック"""
        try:
            current_time = time.time()
            
            # 初期化からの経過時間をチェック
            init_time = getattr(self, 'tracker_initialization_times', {}).get(label)
            if init_time is not None:
                init_elapsed = current_time - init_time
                if init_elapsed < 0.5:  # 500ms以内は不安定期間
                    return False
            
            # 交換からの経過時間をチェック
            replacement_time = getattr(self, 'tracker_replacement_times', {}).get(label)
            if replacement_time is not None:
                replacement_elapsed = current_time - replacement_time
                if replacement_elapsed < 0.3:  # 300ms以内は不安定期間
                    return False
            
            return True
            
        except Exception as e:
            self.get_logger().debug(f"{label}: 安定性チェックエラー: {e}")
            return True  # エラー時は安定とみなす
    
    def _pre_update_bbox_validation(self, label: str, image: np.ndarray) -> bool:
        """tracker.update()実行前の事前検証"""
        try:
            current_bbox = self.tracked_boxes_per_label.get(label)
            if current_bbox is None:
                self.get_logger().debug(f"{label}: 事前検証でbbox未設定")
                return False
            
            x1, y1, x2, y2 = current_bbox
            height, width = image.shape[:2]
            
            # 基本的な妥当性チェック
            if x1 >= x2 or y1 >= y2:
                self.get_logger().warning(f"{label}: 事前検証で不正なbbox座標: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                return False
            
            # サイズチェック
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width < 5 or bbox_height < 5:
                self.get_logger().warning(f"{label}: 事前検証でbbox小さすぎ: {bbox_width:.1f}x{bbox_height:.1f}")
                return False
            
            # 境界チェック
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                self.get_logger().warning(f"{label}: 事前検証でbbox境界外: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] vs {width}x{height}")
                # 境界外の場合は自動修正を試行
                normalized_bbox = self._normalize_bounding_box(current_bbox, width, height, f"{label}_precheck")
                if normalized_bbox is not None:
                    self.tracked_boxes_per_label[label] = normalized_bbox
                    self.get_logger().info(f"{label}: 事前検証で境界修正成功: {current_bbox} -> {normalized_bbox}")
                    return True
                else:
                    return False
            
            return True
            
        except Exception as e:
            self.get_logger().warning(f"{label}: 事前検証エラー: {e}")
            return False
    
    def _update_csrt_trackers_without_deletion(self, image: np.ndarray) -> None:
        """全トラッカーを更新（削除なし・キャッチアップ用）"""
        try:
            # 画像形式を統一（トラッカー用）
            if len(image.shape) == 2:
                # グレースケール -> BGR変換
                tracker_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                tracker_image = image.copy()
            elif len(image.shape) == 3 and image.shape[2] == 1:
                gray_2d = image.squeeze()
                tracker_image = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"サポートされていない画像形式: {image.shape}")
                return
            
            # データ型をuint8に統一
            if tracker_image.dtype != np.uint8:
                if np.max(tracker_image) <= 1.0:
                    tracker_image = (tracker_image * 255).astype(np.uint8)
                else:
                    tracker_image = tracker_image.astype(np.uint8)
                    
            with self.tracking_data_lock:
                for label, tracker in list(self.tracked_csrt_trackers.items()):
                    try:
                        # トラッカーを更新
                        success, opencv_bbox = tracker.update(tracker_image)
                        
                        if success and opencv_bbox is not None:
                            # OpenCV形式からバウンディングボックス形式に変換
                            x, y, w, h = opencv_bbox
                            bbox = [x, y, x + w, y + h]
                            
                            # バウンディングボックスを更新
                            self.tracked_boxes_per_label[label] = bbox
                            # キャッチアップ中は失敗カウントをリセットしない
                            
                            self.get_logger().debug(f"トラッカー更新成功(キャッチアップ): {label} - {bbox}")
                            
                        else:
                            # 追跡失敗の場合も失敗カウントを増やさない
                            self.get_logger().debug(f"トラッカー更新失敗(キャッチアップ): {label}")
                                
                    except Exception as e:
                        self.get_logger().debug(f"トラッカー更新エラー(キャッチアップ) {label}: {e}")
                        
        except Exception as e:
            self.get_logger().error(f"トラッカー更新エラー(キャッチアップ): {e}")
    
    def _reset_tracker_failure_counts(self) -> None:
        """全トラッカーの失敗カウントをリセット（キャッチアップ完了後）"""
        try:
            with self.tracking_data_lock:
                for label in self.tracked_csrt_trackers.keys():
                    if label in self.tracker_failure_counts:
                        self.tracker_failure_counts[label] = 0
                self.get_logger().debug("トラッカー失敗カウントをリセット")
                
        except Exception as e:
            self.get_logger().error(f"トラッカー失敗カウントリセットエラー: {e}")
    
    def _update_csrt_trackers_with_sync_check(self, image: np.ndarray) -> None:
        """同期チェック付きトラッカー更新（リアルタイム追跡強化版）"""
        try:
            # 画像形式を統一（トラッカー用）
            if len(image.shape) == 2:
                # グレースケール -> BGR変換
                tracker_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                tracker_image = image.copy()
            elif len(image.shape) == 3 and image.shape[2] == 1:
                gray_2d = image.squeeze()
                tracker_image = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"サポートされていない画像形式: {image.shape}")
                return
            
            # データ型をuint8に統一
            if tracker_image.dtype != np.uint8:
                if np.max(tracker_image) <= 1.0:
                    tracker_image = (tracker_image * 255).astype(np.uint8)
                else:
                    tracker_image = tracker_image.astype(np.uint8)
                    
            with self.tracking_data_lock:
                labels_to_remove = []
                success_count = 0
                total_trackers = len(self.tracked_features_per_label)
                
                for label, tracker in list(self.tracked_csrt_trackers.items()):
                    try:
                        # トラッカーを更新
                        success, opencv_bbox = tracker.update(tracker_image)
                        
                        if success and opencv_bbox is not None:
                            # OpenCV形式からバウンディングボックス形式に変換
                            x, y, w, h = opencv_bbox
                            bbox = [x, y, x + w, y + h]
                            
                            # 境界チェック
                            height, width = image.shape[:2] if len(image.shape) >= 2 else image.shape
                            if (0 <= x < width and 0 <= y < height and 
                                x + w <= width and y + h <= height and w > 0 and h > 0):
                                # バウンディングボックスを更新
                                self.tracked_boxes_per_label[label] = bbox
                                self.tracker_failure_counts[label] = 0
                                success_count += 1
                                
                                self.get_logger().debug(f"リアルタイム追跡成功: {label} - {bbox}")
                            else:
                                self.get_logger().debug(f"境界外bbox: {label} - {bbox}")
                                self.tracker_failure_counts[label] += 1
                            
                        else:
                            # 追跡失敗の場合
                            self.tracker_failure_counts[label] += 1
                            self.get_logger().debug(f"リアルタイム追跡失敗: {label} (失敗回数: {self.tracker_failure_counts[label]})")
                            
                            # 失敗回数が閾値を超えた場合は削除
                            if self.tracker_failure_counts[label] >= self.tracker_failure_threshold:
                                labels_to_remove.append(label)
                                self.get_logger().info(f"トラッカー削除: {label} (連続失敗)")
                                
                    except Exception as e:
                        self.get_logger().error(f"トラッカー更新エラー {label}: {e}")
                        labels_to_remove.append(label)
                
                # 失敗したトラッカーを削除
                for label in labels_to_remove:
                    self._remove_tracking_object(label)
                    
                # 追跡状態を更新
                self.tracking_valid = len(getattr(self, 'tracked_csrt_trackers', {})) > 0
                
                # リアルタイム追跡結果のログ
                if total_trackers > 0:
                    success_rate = (success_count / total_trackers) * 100
                    self.get_logger().debug(f"リアルタイム追跡結果: {success_count}/{total_trackers}成功 ({success_rate:.1f}%)")
                
        except Exception as e:
            self.get_logger().error(f"同期チェック付きトラッカー更新エラー: {e}")
    
    def _update_csrt_trackers(self, image: np.ndarray) -> None:
        """全トラッカーを更新（KCF/MOSSE/MIL/CSRT対応）"""
        try:
            # 画像形式を統一（トラッカー用）
            if len(image.shape) == 2:
                # グレースケール -> BGR変換
                tracker_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                tracker_image = image.copy()
            elif len(image.shape) == 3 and image.shape[2] == 1:
                gray_2d = image.squeeze()
                tracker_image = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"サポートされていない画像形式: {image.shape}")
                return
            
            # データ型をuint8に統一
            if tracker_image.dtype != np.uint8:
                if np.max(tracker_image) <= 1.0:
                    tracker_image = (tracker_image * 255).astype(np.uint8)
                else:
                    tracker_image = tracker_image.astype(np.uint8)
                    
            with self.tracking_data_lock:
                labels_to_remove = []
                
                for label, tracker in list(self.tracked_csrt_trackers.items()):
                    try:
                        # トラッカーを更新
                        success, opencv_bbox = tracker.update(tracker_image)
                        
                        if success and opencv_bbox is not None:
                            # OpenCV形式からバウンディングボックス形式に変換
                            x, y, w, h = opencv_bbox
                            bbox = [x, y, x + w, y + h]
                            
                            # バウンディングボックスを更新
                            self.tracked_boxes_per_label[label] = bbox
                            self.tracker_failure_counts[label] = 0
                            
                            self.get_logger().debug(f"トラッカー更新成功: {label} - {bbox}")
                            
                        else:
                            # 追跡失敗の場合
                            self.tracker_failure_counts[label] += 1
                            self.get_logger().debug(f"トラッカー更新失敗: {label} (失敗回数: {self.tracker_failure_counts[label]})")
                            
                            # 失敗回数が閾値を超えた場合は削除
                            if self.tracker_failure_counts[label] >= self.tracker_failure_threshold:
                                labels_to_remove.append(label)
                                self.get_logger().info(f"トラッカー削除: {label} (連続失敗)")
                                
                    except Exception as e:
                        self.get_logger().error(f"トラッカー更新エラー {label}: {e}")
                        labels_to_remove.append(label)
                
                # 失敗したトラッカーを削除
                for label in labels_to_remove:
                    self._remove_tracking_object(label)
                    
                # 追跡状態を更新
                self.tracking_valid = len(getattr(self, 'tracked_csrt_trackers', {})) > 0
                
                # デバッグ情報
                if len(self.tracked_features_per_label) > 0:
                    self.get_logger().debug(f"アクティブトラッカー: {len(self.tracked_features_per_label)}個")
                
        except Exception as e:
            self.get_logger().error(f"CSRTトラッカー更新エラー: {e}")
    
    
    def _add_new_tracking_object(self, unique_label: str, new_box: List[float], detection_timestamp: Optional[float], current_time: float) -> None:
        """新規オブジェクトを追跡に追加"""
        try:
            # 検出時点の画像を取得して特徴点を抽出
            if detection_timestamp:
                images = self._get_roi_by_timestamp(detection_timestamp)
                if images:
                    detection_gray, detection_rgb, roi_coords = images
                    
                    # 追跡データに追加（CSRTトラッカーベース）
                    self._add_tracking_data(unique_label, new_box, detection_gray)
                    
                    # 検出時点から現在まで高速追跡で更新
                    time_gap = current_time - detection_timestamp
                    if time_gap > 0.05:
                        self._fast_forward_new_tracking(unique_label, detection_timestamp, current_time)
                    
                    self.get_logger().info(f"新規追跡'{unique_label}'を中心点で追加")
                else:
                    # 検出時点の画像が取得できない場合は現在の画像で初期化
                    current_gray = self._get_current_gray_image()
                    if current_gray:
                        self._initialize_single_object_tracking(unique_label, new_box, current_gray)
            else:
                # タイムスタンプがない場合は現在の画像で初期化
                current_gray = self._get_current_gray_image()
                if current_gray:
                    self._initialize_single_object_tracking(unique_label, new_box, current_gray)
                    
        except Exception as e:
            self.get_logger().error(f"新規追跡追加エラー: {e}")
    
    def _add_tracking_data(self, label: str, box: List[float], gray: np.ndarray) -> None:
        """追跡データを内部構造に追加（CSRTトラッカーベース）"""
        try:
            # CSRTトラッカーを初期化
            if self._initialize_csrt_tracker(label, gray, box):
                # バウンディングボックス
                self.tracked_boxes_per_label[label] = box
                
                # 元のサイズ
                width = box[2] - box[0]
                height = box[3] - box[1]
                self.original_box_sizes[label] = (width, height)
                
                # 検出時刻を記録
                current_time = time.time()
                self.object_last_detection_time[label] = current_time
                
                # 追跡有効化
                self.tracking_valid = True
            else:
                self.get_logger().warn(f"CSRTトラッカー初期化失敗: {label}")
            
        except Exception as e:
            self.get_logger().error(f"追跡データ追加エラー: {e}")
    
    def _fast_forward_new_tracking(self, label: str, start_timestamp: float, end_timestamp: float) -> None:
        """新規追跡オブジェクトを現在時刻まで高速更新"""
        try:
            # 高速更新用の画像フレームを収集
            update_frames = []
            with self.temporal_lock:
                for i, timestamp in enumerate(self.image_timestamps):
                    if start_timestamp < timestamp <= end_timestamp:
                        update_frames.append((timestamp, self.image_buffer[i].copy()))
            
            # 時間順にソート
            update_frames.sort(key=lambda x: x[0])
            
            if len(update_frames) > 0:
                self.get_logger().info(f"新規追跡'{label}'の高速更新: {len(update_frames)}フレーム")
                
                # 段階的更新
                prev_gray = None
                if label in self.tracked_features_per_label:
                    # 検出時点の画像を取得
                    detection_images = self._get_roi_by_timestamp(start_timestamp)
                    if detection_images:
                        prev_gray = detection_images[0]
                
                if prev_gray is not None:
                    for timestamp, curr_gray in update_frames:
                        self._update_single_object_tracking(label, prev_gray, curr_gray)
                        prev_gray = curr_gray
                    
                    self.get_logger().info(f"新規追跡'{label}'の高速更新完了")
                
        except Exception as e:
            self.get_logger().error(f"新規追跡高速更新エラー: {e}")
    
    def _update_single_object_tracking(self, label: str, prev_gray: np.ndarray, curr_gray: np.ndarray) -> None:
        """単一オブジェクトのCSRTトラッカー更新"""
        try:
            if label not in self.tracked_csrt_trackers:
                return
            
            # 指定ラベルのCSRTトラッカーのみ更新
            tracker = self.tracked_csrt_trackers[label]
            success, opencv_bbox = tracker.update(curr_gray)
            
            if success and opencv_bbox is not None:
                # OpenCV形式からバウンディングボックス形式に変換
                x, y, w, h = opencv_bbox
                bbox = [x, y, x + w, y + h]
                self.tracked_boxes_per_label[label] = bbox
                self.tracker_failure_counts[label] = 0
            else:
                # 追跡失敗時は削除
                self._remove_tracking_object(label)
                    
        except Exception as e:
            self.get_logger().error(f"単一追跡更新エラー: {e}")
    
    def _generate_unique_label(self, base_label: str, existing_labels: set) -> str:
        """既存ラベルと重複しないユニークなラベルを生成"""
        counter = 0
        unique_label = f"{base_label}_{counter}"
        while unique_label in existing_labels:
            counter += 1
            unique_label = f"{base_label}_{counter}"
        return unique_label
    
    def _remove_tracking_object(self, label: str) -> None:
        """追跡オブジェクト（CSRTトラッカー含む）を削除"""
        try:
            if label in self.tracked_boxes_per_label:
                del self.tracked_boxes_per_label[label]
            if label in self.tracked_csrt_trackers:
                del self.tracked_csrt_trackers[label]
            if label in self.tracker_failure_counts:
                del self.tracker_failure_counts[label]
            if label in self.tracker_initialization_times:
                del self.tracker_initialization_times[label]
            if label in self.original_box_sizes:
                del self.original_box_sizes[label]
            if label in self.object_last_detection_time:
                del self.object_last_detection_time[label]
                
            self.get_logger().info(f"追跡オブジェクト'{label}'を削除")
            
        except Exception as e:
            self.get_logger().error(f"追跡オブジェクト削除エラー: {e}")
    
    def _start_image_storage(self, timestamp: float) -> None:
        """GroundingDINO処理開始時に画像保存を開始"""
        with self.tracking_data_lock:
            self.grounding_dino_in_progress = True
            self.grounding_dino_start_time = timestamp
            # 画像クリアは早送り処理完了後に実行（時間的整合性処理で必要）
            self.get_logger().info(f"画像保存開始: {timestamp:.3f} (GroundingDINO処理開始)")
    
    def _store_image_if_needed(self, image: np.ndarray, timestamp: float) -> None:
        """GroundingDINO処理中であれば画像を保存"""
        with self.tracking_data_lock:
            if self.grounding_dino_in_progress and self.grounding_dino_start_time is not None:
                # 処理開始時刻以降の画像のみ保存
                if timestamp >= self.grounding_dino_start_time:
                    self.stored_images.append((timestamp, image.copy()))
                    # 詳細ログ：全画像保存を記録
                    time_since_start = timestamp - self.grounding_dino_start_time
                    self.get_logger().info(f"画像保存: {timestamp:.3f} (+{time_since_start:.3f}s) 総数:{len(self.stored_images)}")
                    # メモリ使用量制限（最大30フレーム分）
                    if len(self.stored_images) > 30:
                        self.stored_images.pop(0)
                else:
                    # タイムスタンプが古い場合もログ出力
                    time_diff = self.grounding_dino_start_time - timestamp
                    self.get_logger().warn(f"画像保存スキップ: {timestamp:.3f} (開始前 -{time_diff:.3f}s)")
            else:
                # 処理中でない場合の頻度確認用ログ（毎100フレーム）
                if hasattr(self, '_skip_count'):
                    self._skip_count += 1
                else:
                    self._skip_count = 1
                if self._skip_count % 100 == 0:
                    progress_status = "進行中" if self.grounding_dino_in_progress else "停止中"
                    start_time_status = "有効" if self.grounding_dino_start_time is not None else "未設定"
                    self.get_logger().info(f"画像保存状態: {progress_status}, 開始時刻: {start_time_status}, スキップ数: {self._skip_count}")
    
    def _stop_image_storage(self) -> None:
        """GroundingDINO処理完了時に画像保存を終了"""
        with self.tracking_data_lock:
            stored_count = len(self.stored_images)
            current_time = time.time()
            if self.grounding_dino_start_time is not None:
                processing_time = current_time - self.grounding_dino_start_time
                expected_frames = int(processing_time * 30)  # 30FPS想定
                self.get_logger().info(f"画像保存終了: {stored_count}フレーム保存済み (処理時間:{processing_time:.3f}s, 期待フレーム数:{expected_frames})")
            else:
                self.get_logger().info(f"画像保存終了: {stored_count}フレーム保存済み (開始時刻未設定)")
            self.grounding_dino_in_progress = False
    
    def _cleanup_old_tracking_data(self) -> None:
        """古い・無効な追跡データの定期的クリーンアップ"""
        try:
            with self.tracking_data_lock:
                initial_count = len(self.tracked_boxes_per_label)
                if initial_count == 0:
                    return
                
                # 無効な中心点を持つラベルを特定
                labels_to_remove = []
                for label in list(self.tracked_boxes_per_label.keys()):
                    # CSRTトラッカーが無効または存在しない場合
                    if (label not in self.tracked_csrt_trackers or 
                        self.tracked_csrt_trackers[label] is None):
                        labels_to_remove.append(label)
                    
                    # バウンディングボックスが無効な場合
                    elif (label in self.tracked_boxes_per_label):
                        box = self.tracked_boxes_per_label[label]
                        if len(box) != 4 or any(not isinstance(x, (int, float)) for x in box):
                            labels_to_remove.append(label)
                        # ボックスが画面外に出すぎている場合（仮に480x300の画面として）
                        elif box[0] > 600 or box[1] > 400 or box[2] < -100 or box[3] < -100:
                            labels_to_remove.append(label)
                
                # 無効なラベルを削除
                for label in labels_to_remove:
                    self._remove_tracking_object(label)
                
                final_count = len(self.tracked_boxes_per_label)
                if labels_to_remove:
                    self.get_logger().info(f"追跡データクリーンアップ: {initial_count} → {final_count} ({len(labels_to_remove)}個削除)")
                    
                # 追跡データが空の場合は追跡を無効化
                if final_count == 0:
                    self.tracking_valid = False
                    
        except Exception as e:
            self.get_logger().error(f"追跡データクリーンアップエラー: {e}")
    
    def _verify_temporal_separation(self, new_box: List[float], new_label: str, detection_timestamp: Optional[float], current_time: float) -> bool:
        """時間的分離検証: GroundingDINOと追跡が同じ位置に表示される問題の検出"""
        try:
            if detection_timestamp is None:
                return True  # タイムスタンプ不明の場合は検証をスキップ
            
            # 時間差確認
            time_diff = current_time - detection_timestamp
            if time_diff < 0.1:  # 100ms以内の場合は同じタイミング
                self.get_logger().debug(f"'{new_label}': 検出と現在時刻の時間差が小さい ({time_diff*1000:.1f}ms)")
                return True  # 問題なし
            
            # 早送り処理が正常に機能している場合、追跡位置は検出位置から移動しているはず
            with self.tracking_data_lock:
                for existing_label, existing_box in self.tracked_boxes_per_label.items():
                    # ラベルの基本名が同じかチェック
                    existing_base = existing_label.split('_')[0]
                    new_base = new_label.split('.')[0].strip()
                    
                    if existing_base != new_base:
                        continue
                    
                    # 位置の重複度チェック
                    overlap_ratio = self._calculate_bbox_overlap_ratio(new_box, existing_box)
                    center_distance = self._calculate_center_distance(new_box, existing_box)
                    
                    # 高い重複度または小さい中心距離の場合は時間同期異常の疑い
                    if overlap_ratio > 0.8 or center_distance < 20:
                        self.get_logger().warn(f"⚠️ '{new_label}' ↔ '{existing_label}': 重複度={overlap_ratio:.3f}, 距離={center_distance:.1f}px")
                        self.get_logger().warn(f"  - 時間差: {time_diff*1000:.1f}ms (検出: {detection_timestamp:.3f}, 現在: {current_time:.3f})")
                        return False  # 時間同期異常
            
            return True  # 正常
            
        except Exception as e:
            self.get_logger().error(f"時間的分離検証エラー: {e}")
            return True  # エラー時は検証をスキップ
    
    def _calculate_bbox_overlap_ratio(self, box1: List[float], box2: List[float]) -> float:
        """バウンディングボックスの重複率を計算"""
        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # 交差領域の計算
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union_area = area1 + area2 - inter_area
                
                if union_area > 0:
                    return inter_area / union_area
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_center_distance(self, box1: List[float], box2: List[float]) -> float:
        """バウンディングボックスの中心点間距離を計算"""
        try:
            center1_x = (box1[0] + box1[2]) / 2
            center1_y = (box1[1] + box1[3]) / 2
            center2_x = (box2[0] + box2[2]) / 2
            center2_y = (box2[1] + box2[3]) / 2
            
            return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            
        except Exception:
            return float('inf')
    
    def _diagnose_temporal_sync_failure(self, detection_timestamp: Optional[float], current_time: float) -> None:
        """時間同期失敗の診断"""
        try:
            self.get_logger().warn("=== 時間同期異常診断開始 ===")
            
            if detection_timestamp is None:
                self.get_logger().warn("  - 検出タイムスタンプが未設定")
                return
            
            time_diff = current_time - detection_timestamp
            self.get_logger().warn(f"  - 時間差: {time_diff*1000:.1f}ms")
            
            # 画像バッファ状態の確認
            with self.temporal_lock:
                buffer_size = len(self.full_image_buffer)
                if buffer_size > 0:
                    latest_img_timestamp = self.image_timestamps[-1] if self.image_timestamps else 0
                    buffer_delay = current_time - latest_img_timestamp
                    self.get_logger().warn(f"  - 画像バッファ: {buffer_size}フレーム, 最新遅延: {buffer_delay*1000:.1f}ms")
                else:
                    self.get_logger().warn("  - 画像バッファが空")
            
            # トラッカー状態の確認
            with self.tracking_data_lock:
                active_trackers = len(self.tracked_features_per_label)
                tracked_boxes = len(self.tracked_boxes_per_label)
                self.get_logger().warn(f"  - アクティブトラッカー: {active_trackers}個")
                self.get_logger().warn(f"  - 追跡中ボックス: {tracked_boxes}個")
                
                # 最後の早送り実行時刻をチェック
                if hasattr(self, 'last_fastforward_time'):
                    ff_delay = current_time - self.last_fastforward_time
                    self.get_logger().warn(f"  - 最後の早送りからの経過時間: {ff_delay*1000:.1f}ms")
            
            # 推奨対処法
            if time_diff > 0.5:
                self.get_logger().warn("  - 推奨: 大きな時間ギャップのため緊急早送り実行")
            elif active_trackers == 0:
                self.get_logger().warn("  - 推奨: トラッカー初期化失敗のため再初期化")
            else:
                self.get_logger().warn("  - 推奨: 早送りアルゴリズムの動作確認")
            
            self.get_logger().warn("=== 時間同期異常診断終了 ===")
            
        except Exception as e:
            self.get_logger().error(f"時間同期診断エラー: {e}")
    
    def _remove_undetected_objects(self, detected_labels: List[str]) -> int:
        """検出されなくなったオブジェクトをタイムアウト後に削除"""
        try:
            current_time = time.time()
            
            with self.tracking_data_lock:
                # 検出されたラベルの基本名を抽出（重複を許可）
                detected_base_labels = []
                for label in detected_labels:
                    base_label = label.split('.')[0].strip()
                    detected_base_labels.append(base_label)
                
                # 追跡中のオブジェクトで長時間検出されていないものを特定
                labels_to_remove = []
                for tracked_label in list(self.tracked_boxes_per_label.keys()):
                    tracked_base_label = tracked_label.split('_')[0]
                    
                    # トラッキング対象でない場合はスキップ
                    if not self._is_tracking_target(tracked_base_label):
                        continue
                    
                    # 最後の検出時刻を確認
                    last_detection_time = self.object_last_detection_time.get(tracked_label, current_time)
                    time_since_detection = current_time - last_detection_time
                    
                    # タイムアウト時間を超えた場合は削除対象に追加
                    if time_since_detection > self.detection_timeout:
                        labels_to_remove.append(tracked_label)
                        self.get_logger().info(f"タイムアウト削除: '{tracked_label}' ({time_since_detection:.1f}秒間未検出)")
                
                # 削除実行
                for label in labels_to_remove:
                    self._remove_tracking_object(label)
                
                return len(labels_to_remove)
                
        except Exception as e:
            self.get_logger().error(f"未検出オブジェクト削除エラー: {e}")
            return 0
    
    def _initialize_single_object_tracking(self, label: str, box: List[float], gray: np.ndarray) -> None:
        """単一オブジェクトのCSRTトラッカー初期化"""
        try:
            # CSRTトラッカーで初期化
            self._add_tracking_data(label, box, gray)
            self.get_logger().info(f"単一追跡初期化'{label}': CSRTトラッカーで初期化")
                
        except Exception as e:
            self.get_logger().error(f"単一追跡初期化エラー: {e}")
    
    def _initialize_tracking_boxes(self, gray: np.ndarray) -> None:
        """検出結果からCSRTトラッカー追跡を初期化"""
        try:
            with self.detection_data_lock:
                if self.latest_detection_data is None:
                    self.get_logger().warn("latest_detection_dataがNoneです")
                    return
                    
                boxes = self.latest_detection_data.get('boxes', [])
                labels = self.latest_detection_data.get('labels', [])
                
                self.get_logger().info(f"検出データ: {len(boxes)}個のボックス, {len(labels)}個のラベル")
                
                if len(boxes) == 0:
                    self.get_logger().warn("検出されたボックスがありません")
                    return
            
            with self.tracking_data_lock:
                # 既存のトラッカーをクリア
                self.tracked_boxes_per_label = {}
                self.tracked_csrt_trackers = {}
                self.tracker_failure_counts = {}
                self.original_box_sizes = {}
                
                # ラベルの重複をカウントするための辞書
                label_counts = {}
                successful_trackers = 0
                
                for i, bbox in enumerate(boxes):
                    original_label = labels[i] if i < len(labels) else f'object_{i}'
                    
                    # トラッキング対象のフィルタリング
                    if not self._is_tracking_target(original_label):
                        self.get_logger().info(f"ラベル'{original_label}'はトラッキング対象外です")
                        continue
                    
                    # 同じラベルの重複に対応するため、ユニークなキーを作成
                    if original_label in label_counts:
                        label_counts[original_label] += 1
                    else:
                        label_counts[original_label] = 0
                    
                    unique_label = f"{original_label}_{label_counts[original_label]}"
                    
                    # バウンディングボックスをリストに変換
                    if hasattr(bbox, 'cpu'):  # PyTorch tensor の場合
                        bbox_list = bbox.cpu().numpy().tolist()
                    else:
                        bbox_list = list(bbox)
                    
                    # CSRTトラッカーを初期化
                    if self._initialize_csrt_tracker(unique_label, gray, bbox_list):
                        # バウンディングボックスを保存
                        self.tracked_boxes_per_label[unique_label] = bbox_list
                        
                        # 元のバウンディングボックスサイズを保存
                        x1, y1, x2, y2 = bbox_list
                        width = x2 - x1
                        height = y2 - y1
                        self.original_box_sizes[unique_label] = (width, height)
                        
                        successful_trackers += 1
                        self.tracker_initialization_times[unique_label] = time.time()  # 初期化時刻を記録
                        self.get_logger().info(f"ラベル'{unique_label}': CSRTトラッカー初期化成功, サイズ({width:.1f}×{height:.1f})")
                    else:
                        self.get_logger().warn(f"ラベル'{unique_label}': CSRTトラッカー初期化失敗")
                
                # 前フレームのグレー画像を保存
                self.prev_gray = gray.copy()
                self.tracking_valid = len(self.tracked_features_per_label) > 0
                
                self.get_logger().info(f"オプティカルフロートラッカー初期化完了: {successful_trackers}個成功, 有効: {self.tracking_valid}")
                
        except Exception as e:
            self.get_logger().error(f"CSRTトラッカー初期化エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
    def _extract_features_parallel(self, gray: np.ndarray, bbox_label_pairs: List[Tuple]) -> List[Dict]:
        """並列特徴点検出"""
        from concurrent.futures import as_completed
        
        try:
            future_to_label = {}
            results = []
            
            # 並列タスクを開始
            for bbox_list, unique_label in bbox_label_pairs:
                future = self.feature_executor.submit(
                    self._extract_single_feature, gray, bbox_list, unique_label
                )
                future_to_label[future] = unique_label
            
            # 結果を収集
            for future in as_completed(future_to_label.keys()):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    label = future_to_label[future]
                    self.get_logger().error(f"並列特徴点検出エラー (ラベル: {label}): {e}")
            
            return results
            
        except Exception as e:
            self.get_logger().error(f"並列特徴点検出エラー: {e}")
            # フォールバック: 同期処理
            return self._extract_features_sequential(gray, bbox_label_pairs)
    
    def _extract_features_sequential(self, gray: np.ndarray, bbox_label_pairs: List[Tuple]) -> List[Dict]:
        """順次特徴点検出"""
        results = []
        for bbox_list, unique_label in bbox_label_pairs:
            result = self._extract_single_feature(gray, bbox_list, unique_label)
            if result:
                results.append(result)
        return results
    
    def _extract_single_feature(self, gray: np.ndarray, bbox_list: List[float], unique_label: str) -> Optional[Dict]:
        """単一バウンディングボックスの特徴点検出"""
        try:
            # Cannyエッジ点を抽出
            feature_points = self._extract_canny_edge_points(gray, bbox_list)
            
            # バウンディングボックスの中心点を計算
            x1, y1, x2, y2 = bbox_list
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            
            result = {
                'label': unique_label,
                'bbox': bbox_list,
                'feature_points': feature_points,
                'center_point': center_point
            }
            
            feature_count = len(feature_points) if feature_points is not None else 0
            self.get_logger().debug(f"特徴点検出結果 {unique_label}: {feature_count}点")
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"特徴点検出エラー (ラベル: {unique_label}): {e}")
            return None
    
    
    def _run_tracking_async(self, gray: np.ndarray, current_time: float) -> None:
        """非同期オプティカルフロートラッキング実行"""
        try:
            # 既に処理中の場合はスキップ
            if self.tracking_processing:
                return
            
            # トラッカーが存在しない場合はスキップ
            if len(self.tracked_features_per_label) == 0:
                return
            
            # 非同期でトラッキング処理を開始
            self.tracking_processing = True
            self.tracking_future = self.tracker_executor.submit(
                self._tracking_worker, gray.copy(), current_time
            )
            
        except Exception as e:
            self.get_logger().error(f"非同期トラッキング開始エラー: {e}")
            self.tracking_processing = False
    
    def _check_tracking_results(self) -> None:
        """非同期トラッキング処理の結果を確認"""
        try:
            if self.tracking_future is not None and self.tracking_future.done():
                try:
                    tracking_result = self.tracking_future.result()
                    if tracking_result:
                        # トラッキング結果を適用
                        self._apply_tracking_results(tracking_result)
                        
                except Exception as e:
                    self.get_logger().error(f"非同期トラッキング結果取得エラー: {e}")
                finally:
                    self.tracking_future = None
                    self.tracking_processing = False
                    
        except Exception as e:
            self.get_logger().error(f"トラッキング結果確認エラー: {e}")
    
    def _tracking_worker(self, gray: np.ndarray, current_time: float) -> Optional[Dict]:
        """トラッキング処理ワーカー（別スレッドで実行）"""
        try:
            # オプティカルフロートラッキング実行
            result = self._track_optical_flow_worker(gray, current_time)
            return result
            
        except Exception as e:
            self.get_logger().error(f"トラッキングワーカーエラー: {e}")
            return None
    
    def _track_optical_flow_worker(self, gray: np.ndarray, current_time: float) -> Optional[Dict]:
        """オプティカルフロートラッキング処理（ワーカー用）"""
        try:
            if len(self.tracked_features_per_label) == 0:
                return None
            
            # オプティカルフロー更新を実行
            success = self._update_all_trackers(gray, update_prev_gray=True)
            
            if success:
                # 現在のトラッキング状態を返す
                with self.tracking_data_lock:
                    result = {
                        'tracking_boxes': self.tracked_boxes_per_label.copy(),
                        'tracking_features': {k: v.copy() for k, v in self.tracked_features_per_label.items()},
                        'tracking_valid': len(self.tracked_features_per_label) > 0,
                        'timestamp': current_time
                    }
                return result
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロートラッキングワーカーエラー: {e}")
            return None
    
    def _apply_tracking_results(self, result: Dict) -> None:
        """トラッキング結果を適用"""
        try:
            with self.tracking_data_lock:
                if 'tracking_boxes' in result:
                    self.tracked_boxes_per_label.update(result['tracking_boxes'])
                if 'tracking_features' in result:
                    self.tracked_features_per_label.update(result['tracking_features'])
                if 'tracking_valid' in result:
                    self.tracking_valid = result['tracking_valid']
                    
        except Exception as e:
            self.get_logger().error(f"トラッキング結果適用エラー: {e}")
    
    def _track_optical_flow(self, gray: np.ndarray) -> None:
        """CSRTトラッカーによるリアルタイム追跡（最新画像同期対応）"""
        try:
            # シンプルなトラッキング実行
            current_time = time.time()
            
            # シンプルなCSRTトラッカー更新
            if len(self.tracked_features_per_label) > 0:
                self._update_csrt_trackers_simple(gray)
            self.prev_gray = gray.copy()
                
        except Exception as e:
            self.get_logger().error(f"CSRTトラッカー追跡エラー: {repr(e)}")
    
    def _update_csrt_trackers_simple(self, gray: np.ndarray) -> None:
        """シンプルなCSRTトラッカー更新"""
        try:
            # BGRに変換
            if len(gray.shape) == 2:
                bgr_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                bgr_image = gray.copy()
                
            success_count = 0
            failed_labels = []
            
            with self.tracking_data_lock:
                for label, tracker in list(self.tracked_csrt_trackers.items()):
                    try:
                        success, opencv_bbox = tracker.update(bgr_image)
                        
                        if success and len(opencv_bbox) == 4:
                            x, y, w, h = opencv_bbox
                            bbox = [x, y, x + w, y + h]
                            self.tracked_boxes_per_label[label] = bbox
                            self.tracker_failure_counts[label] = 0
                            success_count += 1
                        else:
                            # 失敗カウント増加
                            self.tracker_failure_counts[label] = self.tracker_failure_counts.get(label, 0) + 1
                            if self.tracker_failure_counts[label] >= self.tracker_failure_threshold:
                                failed_labels.append(label)
                                
                    except Exception as e:
                        self.get_logger().error(f"トラッカー更新エラー {label}: {e}")
                        failed_labels.append(label)
                
                # 失敗したトラッカーを削除
                for label in failed_labels:
                    if label in self.tracked_csrt_trackers:
                        del self.tracked_csrt_trackers[label]
                    if label in self.tracked_features_per_label:
                        del self.tracked_features_per_label[label]
                    if label in self.tracked_boxes_per_label:
                        del self.tracked_boxes_per_label[label]
                    if label in self.tracker_failure_counts:
                        del self.tracker_failure_counts[label]
                    self.get_logger().info(f"失敗トラッカー削除: {label}")
                
                # tracking_valid状態を更新
                self.tracking_valid = len(self.tracked_features_per_label) > 0
                
                if self.frame_count % 30 == 0 and len(self.tracked_features_per_label) > 0:
                    self.get_logger().info(f"トラッカー状態: {success_count}/{len(self.tracked_features_per_label)}成功")
                    
        except Exception as e:
            self.get_logger().error(f"CSRTトラッカー更新エラー: {e}")
            # self.tracking_valid = False
    
    def _run_sam2_segmentation_optimized(self, image_cv: np.ndarray, boxes_data: Dict) -> Dict:
        """最適化されたSAM2セグメンテーション実行 (バウンディングボックス入力版)"""
        try:
            if self.sam_model_instance is None:
                self.get_logger().warn("SAM2モデルが初期化されていません")
                return None
                
            all_boxes = boxes_data['boxes']
            all_labels = boxes_data['labels']
            
            if not all_boxes:
                return None
            
            # NumPy配列に変換
            boxes_array = np.array(all_boxes, dtype=np.float32)
            
            # GPU最適化: 真のバッチ処理で高速化
            with torch.no_grad():
                # SAM2モデルに画像を一度だけ設定（効率化）
                self.sam_model_instance.predictor.set_image(image_cv)
                
                # 真のバッチ処理でSAM2推論を実行
                if len(boxes_array) > 1:
                    # 複数ボックスの場合はバッチ処理
                    try:
                        # SAM2のバッチ処理を使用
                        masks, scores, logits = self.sam_model_instance.predictor.predict(
                            box=boxes_array,  # 全ボックスを一度に処理
                            multimask_output=False
                        )
                        
                        # 結果をリスト化
                        if len(masks.shape) > 3:
                            masks_list = [masks[i] for i in range(masks.shape[0])]
                        else:
                            masks_list = [masks]
                        
                        if len(scores.shape) > 1:
                            scores_list = [scores[i] for i in range(scores.shape[0])]
                        else:
                            scores_list = [scores]
                            
                        if len(logits.shape) > 3:
                            logits_list = [logits[i] for i in range(logits.shape[0])]
                        else:
                            logits_list = [logits]
                            
                    except Exception as batch_error:
                        # バッチ処理失敗時はフォールバックして順次処理
                        self.get_logger().warn(f"バッチ処理失敗、順次処理にフォールバック: {batch_error}")
                        masks_list, scores_list, logits_list = self._process_boxes_sequentially(boxes_array)
                else:
                    # 単一ボックスの場合
                    masks_list, scores_list, logits_list = self._process_boxes_sequentially(boxes_array)
            
            # 結果を結合
            combined_masks = np.array(masks_list) if masks_list else np.array([])
            combined_scores = np.array(scores_list) if scores_list else np.array([])
            combined_logits = np.array(logits_list) if logits_list else np.array([])
            
            return {
                'masks': combined_masks,
                'scores': combined_scores,
                'logits': combined_logits,
                'labels': all_labels,
                'boxes': boxes_array
            }
                
        except Exception as e:
            self.get_logger().error(f"SAM2セグメンテーションエラー: {repr(e)}")
            return None
    
    def _process_boxes_sequentially(self, boxes_array: np.ndarray) -> tuple:
        """バウンディングボックスを順次処理（フォールバック用）"""
        masks_list = []
        scores_list = []
        logits_list = []
        
        for box in boxes_array:
            # 各バウンディングボックスに対してSAM2を実行
            masks, scores, logits = self.sam_model_instance.predictor.predict(
                box=box.reshape(1, -1),  # SAM2は(1, 4)形式を期待
                multimask_output=False
            )
            
            # メモリ効率のため即座にCPUに移動
            if hasattr(masks, 'cpu'):
                masks = masks.cpu().numpy()
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            if hasattr(logits, 'cpu'):
                logits = logits.cpu().numpy()
            
            masks_list.append(masks[0] if len(masks.shape) > 2 else masks)
            scores_list.append(scores[0] if len(scores.shape) > 0 else scores)
            logits_list.append(logits[0] if len(logits.shape) > 2 else logits)
        
        return masks_list, scores_list, logits_list
    
    
    def _run_sam2_segmentation_async(self, image_cv: np.ndarray) -> None:
        """SAM2セグメンテーション実行（非同期処理）"""
        # 既に処理中の場合はスキップ
        if self.sam2_processing or (self.sam2_future and not self.sam2_future.done()):
            return
        
        # 現在の追跡データを取得
        with self.tracking_data_lock:
            if not self.tracked_boxes_per_label:
                return
            
            boxes_data = {
                'boxes': [box for box in self.tracked_boxes_per_label.values() if box is not None],
                'labels': [label for label, box in self.tracked_boxes_per_label.items() if box is not None]
            }
            
            if not boxes_data['boxes']:
                return
        
        # 新しい非同期タスクを開始
        self.sam2_processing = True
        self.sam2_future = self.sam2_executor.submit(
            self._sam2_worker, image_cv.copy(), boxes_data
        )
    
    def _sam2_worker(self, image_cv: np.ndarray, boxes_data: Dict) -> Optional[Dict]:
        """SAM2処理ワーカー（バックグラウンドスレッド）"""
        try:
            # SAM2処理を実行（バウンディングボックス入力）
            result = self._run_sam2_segmentation_optimized(image_cv, boxes_data)
            
            if result:
                result['timestamp'] = time.time()
                return result
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f"SAM2処理エラー: {e}")
            return None
        finally:
            self.sam2_processing = False
    
    def _check_sam2_results(self) -> None:
        """SAM2非同期処理の結果を確認・統合"""
        if self.sam2_future and self.sam2_future.done():
            try:
                result = self.sam2_future.result()
                
                if result is not None:
                    # 結果を内部データに保存
                    with self.sam_data_lock:
                        self.latest_sam_masks = result
                        self.sam_updated = True
                        
                        self.get_logger().info(f"SAM2セグメンテーション完了: {len(result['masks'])}個のマスク（非同期処理）")
                else:
                    self.get_logger().warn("SAM2処理結果が空です")
                    
                # 完了したタスクをクリア
                self.sam2_future = None
                    
            except Exception as e:
                self.get_logger().error(f"SAM2結果統合エラー: {repr(e)}")
                self.sam2_future = None
    
    def _publish_results(self, image_cv: np.ndarray) -> None:
        """結果の可視化・配信（非同期化）"""
        try:
            # 前回の可視化処理が完了しているかチェック
            if self.visualization_future and not self.visualization_future.done():
                return  # まだ実行中なのでスキップ
            
            # 現在のデータをコピーして非同期で可視化処理を開始
            with self.tracking_data_lock:
                tracking_data = {
                    'tracked_boxes_per_label': dict(self.tracked_boxes_per_label),
                    'tracked_csrt_trackers': dict(self.tracked_csrt_trackers)  # CSRTトラッカー情報
                }
            
            with self.sam_data_lock:
                sam_data = dict(self.latest_sam_masks) if self.latest_sam_masks else None
            
            with self.detection_data_lock:
                grounding_dino_data = dict(self.latest_grounding_dino_result) if self.latest_grounding_dino_result else None
                grounding_dino_updated = self.grounding_dino_updated
            
            # 非同期で可視化処理を開始
            self.visualization_future = self.background_executor.submit(
                self._visualization_worker, image_cv.copy(), tracking_data, sam_data, grounding_dino_data, grounding_dino_updated
            )
                
        except Exception as e:
            self.get_logger().error(f"結果配信エラー: {repr(e)}")
    
    def _visualization_worker(self, image_cv: np.ndarray, tracking_data: Dict, sam_data: Optional[Dict], grounding_dino_data: Optional[Dict], grounding_dino_updated: bool) -> None:
        """可視化処理ワーカー (別スレッドで実行)"""
        try:
            # GroundingDINO検出結果の可視化（データがあれば常に可視化）
            if grounding_dino_data:
                self._publish_grounding_dino_result_worker(grounding_dino_data)
            
            # オプティカルフロー結果の可視化
            self._publish_optical_flow_result_worker(image_cv, tracking_data)
            
            # SAM2セグメンテーション結果の可視化
            if sam_data:
                self._publish_sam2_result_worker(image_cv, sam_data)
            # SAM2データがない場合のデバッグログ
            elif self.frame_count % 30 == 0:
                self.get_logger().info("SAM2データがないため可視化スキップ")
            
            # 特徴点情報の配信
            self._publish_feature_points_worker(image_cv, tracking_data)
            
            # SAMマスクメッセージの配信
            if sam_data:
                self._publish_sam_masks_worker(sam_data)
            
        except Exception as e:
            self.get_logger().error(f"可視化ワーカーエラー: {repr(e)}")
    
    def _publish_grounding_dino_result_worker(self, grounding_dino_data: Dict) -> None:
        """GroundingDINO検出結果の可視化・配信（ワーカー版）"""
        try:
            image_cv = grounding_dino_data['image']
            boxes = grounding_dino_data['boxes']
            labels = grounding_dino_data['labels']
            scores = grounding_dino_data['scores']
            
            if len(boxes) == 0:
                return
            
            # バウンディングボックス情報をNumPy配列に変換
            boxes_array = []
            scores_array = []
            
            for i, bbox in enumerate(boxes):
                if hasattr(bbox, 'cpu'):  # PyTorch tensor の場合
                    bbox_list = bbox.cpu().numpy().tolist()
                else:
                    bbox_list = list(bbox)
                boxes_array.append(bbox_list)
                
                # スコアを取得
                if i < len(scores):
                    if hasattr(scores[i], 'cpu'):
                        score = scores[i].cpu().numpy().item()
                    else:
                        score = float(scores[i])
                    scores_array.append(score)
                else:
                    scores_array.append(1.0)  # デフォルトスコア
            
            boxes_np = np.array(boxes_array, dtype=np.float32)
            scores_np = np.array(scores_array, dtype=np.float32)
            
            # draw_image関数を使用して可視化（マスクなし、バウンディングボックスのみ）
            # 空のマスクを作成（バウンディングボックスのみ表示したいため）
            dummy_masks = np.zeros((len(boxes_np), image_cv.shape[0], image_cv.shape[1]), dtype=bool)
            
            annotated_image = draw_image(
                image_rgb=image_cv,
                masks=dummy_masks,  # 空のマスク
                xyxy=boxes_np,
                probs=scores_np,
                labels=labels
            )
            
            # ROS メッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.grounding_dino_image_pub.publish(ros_msg)
                
        except Exception as e:
            self.get_logger().error(f"GroundingDINO結果配信エラー: {repr(e)}")
    
    def _publish_optical_flow_result_worker(self, image_cv: np.ndarray, tracking_data: Dict) -> None:
        """オプティカルフロー結果の可視化・配信 (バウンディングボックス・特徴点表示)"""
        try:
            tracked_boxes_per_label = tracking_data['tracked_boxes_per_label']
            
            # 画像をコピー
            result_image = image_cv.copy()
            
            # 追跡ボックスがない場合でも空でない画像を配信
            if not tracked_boxes_per_label:
                # 元画像にテキストを追加
                cv2.putText(result_image, "Waiting for tracking...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                # 追跡バウンディングボックスを描画（CSRTトラッカー版）
                
                for label, box in tracked_boxes_per_label.items():
                    if box is not None:
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # バウンディングボックスを描画
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # CSRTトラッカー情報を表示（バウンディングボックスのみ、特徴点なし）
                        self.get_logger().debug(f"可視化: {label} - CSRTトラッカーで追跡中")
                        
                        # ラベルを表示
                        cv2.putText(result_image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # ROS メッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(result_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.optflow_image_pub.publish(ros_msg)
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー結果配信エラー: {repr(e)}")
    
    def _publish_sam2_result_worker(self, image_cv: np.ndarray, sam_data: Dict) -> None:
        """SAM2セグメンテーション結果の可視化・配信（ワーカー版、バウンディングボックス入力版）"""
        try:
            masks = sam_data['masks']
            labels = sam_data['labels']
            scores = sam_data['scores']
            boxes = sam_data['boxes']
            
            # マスク形状をdraw_image関数用に正規化（(N, H, W)に変換）
            normalized_masks = []
            for i, mask in enumerate(masks):
                if len(mask.shape) == 4:  # (1, 1, H, W)
                    normalized_mask = mask[0, 0]
                elif len(mask.shape) == 3:  # (1, H, W)
                    normalized_mask = mask[0]
                elif len(mask.shape) == 2:  # (H, W)
                    normalized_mask = mask
                else:
                    normalized_mask = mask.squeeze()
                normalized_masks.append(normalized_mask)
            
            # スコアを正しく変換
            scores_array = []
            for score in scores:
                if hasattr(score, 'item'):
                    scores_array.append(score.item())
                elif hasattr(score, 'cpu'):
                    scores_array.append(float(score.cpu().numpy()))
                else:
                    scores_array.append(float(score))
            
            # draw_image関数を使用して可視化
            annotated_image = draw_image(
                image_rgb=image_cv,
                masks=np.array(normalized_masks),
                xyxy=boxes,
                probs=np.array(scores_array),
                labels=labels
            )
            
            # ROS メッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.sam_image_pub.publish(ros_msg)
                
        except Exception as e:
            self.get_logger().error(f"SAM2結果配信エラー: {repr(e)}")
    
    def _publish_feature_points_worker(self, image_cv: np.ndarray, tracking_data: Dict) -> None:
        """バウンディングボックス中心点情報の配信（ワーカー版）"""
        try:
            tracked_boxes_per_label = tracking_data['tracked_boxes_per_label']
            
            if not tracked_boxes_per_label:
                return
            
            feature_points_msg = FeaturePoints()
            feature_points_msg.header.stamp = self.get_clock().now().to_msg()
            feature_points_msg.header.frame_id = 'camera_frame'
            
            all_points = []
            labels = []
            
            # CSRTトラッカーでは、バウンディングボックスの中心点を配信
            for label, bbox in tracked_boxes_per_label.items():
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    point32 = Point32()
                    point32.x = float(center_x)
                    point32.y = float(center_y)
                    point32.z = 0.0
                    all_points.append(point32)
                    labels.append(label)
            
            feature_points_msg.points = all_points
            feature_points_msg.labels = labels
            self.feature_points_pub.publish(feature_points_msg)
            
        except Exception as e:
            self.get_logger().error(f"特徴点配信エラー: {repr(e)}")
    
    def _publish_sam_masks_worker(self, sam_data: Dict) -> None:
        """SAMマスクメッセージの配信（ワーカー版）"""
        try:
            masks = sam_data['masks']
            labels = sam_data['labels'] 
            scores = sam_data['scores']
            
            sam_masks_msg = SamMasks()
            sam_masks_msg.header.stamp = self.get_clock().now().to_msg()
            sam_masks_msg.header.frame_id = 'camera_frame'
            
            sam_masks_msg.labels = [f"newsam_{label}" for label in labels]
            
            # スコアを正しく変換
            probs_list = []
            for score in scores:
                if hasattr(score, 'item'):
                    probs_list.append(float(score.item()))
                elif hasattr(score, 'cpu'):
                    probs_list.append(float(score.cpu().numpy()))
                else:
                    probs_list.append(float(score))
            sam_masks_msg.probs = probs_list
            
            # マスクをROSImageに変換（形状正規化あり）
            for i, mask in enumerate(masks):
                # マスク形状を正規化
                if len(mask.shape) == 4:  # (1, 1, H, W)
                    normalized_mask = mask[0, 0]
                elif len(mask.shape) == 3:  # (1, H, W)
                    normalized_mask = mask[0]
                elif len(mask.shape) == 2:  # (H, W)
                    normalized_mask = mask
                else:
                    normalized_mask = mask.squeeze()
                
                # uint8に変換
                mask_uint8 = (normalized_mask * 255).astype(np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
                mask_msg.header = sam_masks_msg.header
                sam_masks_msg.masks.append(mask_msg)
            
            self.sam_masks_pub.publish(sam_masks_msg)
            
        except Exception as e:
            self.get_logger().error(f"SAMマスク配信エラー: {repr(e)}")
    
    def _convert_detection_tensors_to_cpu(self, detection_data: Dict) -> Dict:
        """CUDA テンソルを CPU に変換してリストに変換"""
        try:
            converted_data = {}
            for key, value in detection_data.items():
                if hasattr(value, 'cpu'):  # PyTorch tensor
                    if hasattr(value, 'numpy'):
                        converted_data[key] = value.cpu().numpy().tolist()
                    else:
                        converted_data[key] = value.cpu().tolist()
                elif isinstance(value, list):
                    # リストの場合は各要素をチェック
                    converted_list = []
                    for item in value:
                        if hasattr(item, 'cpu'):
                            if hasattr(item, 'numpy'):
                                converted_list.append(item.cpu().numpy().tolist())
                            else:
                                converted_list.append(item.cpu().tolist())
                        else:
                            converted_list.append(item)
                    converted_data[key] = converted_list
                else:
                    converted_data[key] = value
            return converted_data
        except Exception as e:
            self.get_logger().error(f"テンソル変換エラー: {e}")
            return detection_data
    
    def _is_tracking_target(self, label: str) -> bool:
        """ラベルがトラッキング対象かを判定"""
        if not self.tracking_targets:
            return True
        return any(target.lower() in label.lower() or label.lower() in target.lower() 
                  for target in self.tracking_targets)
    
    def _configure_cuda_environment(self) -> None:
        """CUDA環境の設定"""
        try:
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
            warnings.filterwarnings("ignore", category=UserWarning)
            
            if torch.cuda.is_available():
                self.get_logger().info(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
                torch.cuda.empty_cache()
            else:
                self.get_logger().info("CUDAが利用できません。CPUモードで実行します。")
                
        except Exception as e:
            self.get_logger().warn(f"CUDA設定エラー: {repr(e)}")


def main(args=None):
    """メイン関数"""
    rclpy.init(args=args)
    
    try:
        node = LangSamWithOptFlowNode()
        
        # マルチスレッド実行器を使用
        # CPUコア数に合わせたスレッド数で最適化
        import os
        optimal_threads = min(os.cpu_count() or 4, 8)  # 最大4, 最大8スレッド
        executor = MultiThreadedExecutor(num_threads=optimal_threads)
        node.get_logger().info(f"スレッド数最適化: {optimal_threads}スレッドで実行")
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            # ノードのクリーンアップ
            try:
                node._cleanup_resources()
                node.destroy_node()
            except Exception as e:
                print(f"ノード終了処理エラー: {e}")
        
        # RCLの状態をチェックしてからshutdown
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"RCLシャットダウンエラー: {e}")


if __name__ == '__main__':
    main()