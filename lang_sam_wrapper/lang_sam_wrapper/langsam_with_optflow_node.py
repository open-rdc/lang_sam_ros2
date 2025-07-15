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
from lang_sam_wrapper.feature_extraction import extract_harris_corners_from_bbox
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
        
        # Harris corner検出パラメータ
        self.harris_max_corners = self.get_config_param('harris_max_corners', 5)
        self.harris_quality_level = self.get_config_param('harris_quality_level', 0.02)
        self.harris_min_distance = self.get_config_param('harris_min_distance', 15)
        self.harris_block_size = self.get_config_param('harris_block_size', 3)
        self.harris_k = self.get_config_param('harris_k', 0.04)
        self.use_harris_detector = self.get_config_param('use_harris_detector', True)
        
        # 特徴点選択パラメータ
        self.feature_selection_method = self.get_config_param('feature_selection_method', 'harris_response')
        
        # オプティカルフローパラメータ
        self.optical_flow_win_size_x = self.get_config_param('optical_flow_win_size_x', 21)
        self.optical_flow_win_size_y = self.get_config_param('optical_flow_win_size_y', 21)
        self.optical_flow_win_size = (self.optical_flow_win_size_x, self.optical_flow_win_size_y)
        self.optical_flow_max_level = self.get_config_param('optical_flow_max_level', 3)
        self.optical_flow_criteria_eps = self.get_config_param('optical_flow_criteria_eps', 0.01)
        self.optical_flow_criteria_max_count = self.get_config_param('optical_flow_criteria_max_count', 15)
        
        # 追跡検証パラメータ
        self.max_displacement = self.get_config_param('max_displacement', 50.0)
        self.min_valid_ratio = self.get_config_param('min_valid_ratio', 0.5)
        
        # エッジ検出パラメータ
        self.canny_low_threshold = self.get_config_param('canny_low_threshold', 50)
        self.canny_high_threshold = self.get_config_param('canny_high_threshold', 150)
        self.min_edge_pixels = self.get_config_param('min_edge_pixels', 10)
        
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
        self.optical_flow_interpolation_threshold = self.get_config_param('optical_flow_interpolation_threshold', 0.1)
        
        # 追跡統合設定
        self.enable_tracking_integration = self.get_config_param('enable_tracking_integration', True)
        self.overlap_iou_threshold = self.get_config_param('overlap_iou_threshold', 0.3)
        self.overlap_distance_threshold = self.get_config_param('overlap_distance_threshold', 100.0)
        self.merge_weight_existing = self.get_config_param('merge_weight_existing', 0.7)
        self.merge_weight_new = self.get_config_param('merge_weight_new', 0.3)
        self.detection_timeout = self.get_config_param('detection_timeout', 2.0)  # 2秒間検出されないと削除
        
        # オプティカルフロー criteria
        self.optical_flow_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.optical_flow_criteria_max_count,
            self.optical_flow_criteria_eps
        )
    
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
            self.prev_gray: Optional[np.ndarray] = None
            self.tracked_boxes_per_label: Dict[str, List[float]] = {}  # バウンディングボックス追跡用
            self.tracked_centers_per_label: Dict[str, np.ndarray] = {}  # 中心点追跡用（互換性のため残す）
            self.tracked_features_per_label: Dict[str, np.ndarray] = {}  # 複数特徴点追跡用
            self.original_box_sizes: Dict[str, Tuple[float, float]] = {}  # 初期バウンディングボックスサイズ (width, height)
            self.feature_to_center_offsets: Dict[str, np.ndarray] = {}  # 特徴点と中心点の相対距離 (N, 2)
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
        
        # 特徴点処理並列化用
        self.feature_executor = ThreadPoolExecutor(max_workers=self.feature_processing_workers, thread_name_prefix="FeatureProcessing")
        
        # 特徴点プール管理
        self.feature_pool = FeaturePool(max_features=100, min_features=20)
        
        # GPUリソース管理
        self.gpu_manager = get_gpu_manager()
        self.gpu_manager.set_memory_threshold(0.8)  # 80%で警告
        self.optical_flow_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OpticalFlow")
        
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
            if hasattr(self, 'optical_flow_executor'):
                self.optical_flow_executor.shutdown(wait=True)
            
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
                elif name.startswith('optical_flow_'):
                    default_value = 21 if 'size' in name else 3
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
            
            # 最適化ログ（頻度を削減）
            if self.frame_count % 100 == 0 and self.frame_count > 0:
                self.get_logger().info(f"フレーム {self.frame_count} 処理中")
            
            # 1. GroundingDINO検出（時間整合性対応版）
            self._check_grounding_dino_results()  # 完了した非同期処理の結果を確認
            should_run_gdino = self._should_run_grounding_dino(current_time)
            if should_run_gdino:
                gdino_start = time.time()
                if self.frame_count % 30 == 0:
                    self.get_logger().info("GroundingDINO検出を非同期実行（時間追跡有効）")
                # 時間追跡付きで検出を開始（実際の画像タイムスタンプを使用）
                request_id = self._run_grounding_dino_detection_with_timestamp(image_cv, current_time)
                self.last_grounding_dino_time = current_time
                self.performance_stats['gdino_times'].append(time.time() - gdino_start)
            else:
                if self.frame_count % 200 == 0:  # ログ頻度を削減
                    time_since_last = current_time - self.last_grounding_dino_time
                    self.get_logger().info(f"GroundingDINOスキップ: {time_since_last:.1f}s/{self.grounding_dino_interval}s")
            
            # 2. 時間的整合性を考慮した追跡初期化処理
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
                    # フォールバック: 従来の初期化方式
                    self._initialize_tracking_boxes(gray)
                    # 従来方式の場合は通常のオプティカルフロー処理を実行
                    self._track_optical_flow(gray)
                    
                self.detection_updated = False
                self.performance_stats['optflow_times'].append(time.time() - optflow_start)
            else:
                # 通常のオプティカルフロー処理（検出更新がない場合）
                self._track_optical_flow(gray)
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
        """指定されたタイムスタンプに最も近いフル画像を取得"""
        try:
            with self.temporal_lock:
                if not hasattr(self, 'full_image_buffer') or len(self.full_image_buffer) == 0:
                    self.get_logger().warn("フル画像バッファが空です")
                    return None
                
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
                
                if closest_index < len(self.full_image_buffer):
                    return self.full_image_buffer[closest_index].copy()
                else:
                    return None
                       
        except Exception as e:
            self.get_logger().error(f"フル画像取得エラー: {e}")
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
                    
                # 特徴点をROI座標系に変換
                features = self.tracked_features_per_label[label]
                
                # 空の配列チェック
                if features.size == 0:
                    self.get_logger().warning(f"     - {label}: 空の特徴点配列")
                    continue
                
                # 特徴点の形状を確認して安全に変換
                if len(features.shape) == 3:  # (N, 1, 2) 形状
                    if features.shape[0] == 0:  # 空の配列
                        self.get_logger().warning(f"     - {label}: 空の特徴点配列")
                        continue
                    roi_features = features.copy()
                    roi_features[:, 0, 0] -= roi_x1
                    roi_features[:, 0, 1] -= roi_y1
                elif len(features.shape) == 2:  # (N, 2) 形状
                    if features.shape[0] == 0:  # 空の配列
                        self.get_logger().warning(f"     - {label}: 空の特徴点配列")
                        continue
                    roi_features = features.copy()
                    roi_features[:, 0] -= roi_x1
                    roi_features[:, 1] -= roi_y1
                    # (N, 1, 2) 形状に変換
                    roi_features = roi_features.reshape(-1, 1, 2)
                else:
                    self.get_logger().error(f"     - {label}: 不正な特徴点形状: {features.shape}")
                    continue
                
                self.get_logger().debug(f"     - {label}: {len(features)}個の特徴点をROI座標系に変換")
                
                # 特徴点がROI境界内にあるかチェック
                if len(roi_features.shape) == 3:
                    x_coords = roi_features[:, 0, 0]
                    y_coords = roi_features[:, 0, 1]
                else:
                    x_coords = roi_features[:, 0]
                    y_coords = roi_features[:, 1]
                
                # 境界チェック
                valid_mask = (
                    (x_coords >= 0) & (x_coords < current_roi_gray.shape[1]) &
                    (y_coords >= 0) & (y_coords < current_roi_gray.shape[0])
                )
                
                if not np.any(valid_mask):
                    self.get_logger().warning(f"     - {label}: 特徴点がROI境界外")
                    continue
                
                # オプティカルフロー実行
                try:
                    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                        detection_roi_gray, current_roi_gray, roi_features, None,
                        winSize=(self.optical_flow_win_size_x, self.optical_flow_win_size_y),
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                except cv2.error as e:
                    self.get_logger().error(f"     - {label}: オプティカルフロー実行エラー: {e}")
                    continue
                
                # 有効な特徴点をフィルタリング
                valid_indices = (status == 1).flatten()
                valid_count = np.sum(valid_indices)
                valid_ratio = valid_count / len(roi_features) if len(roi_features) > 0 else 0
                
                self.get_logger().debug(f"     - {label}: {valid_count}/{len(roi_features)}個有効 ({valid_ratio*100:.1f}%)")
                
                if valid_ratio < 0.5:  # 50%以下の場合は削除
                    self.get_logger().debug(f"     - {label}: 有効率低いため削除")
                    self._remove_tracking_object(label)
                    continue
                
                # フル座標系に戻す（形状を考慮）
                valid_features = curr_features[valid_indices]
                if len(valid_features.shape) == 3:  # (N, 1, 2) 形状
                    valid_features[:, 0, 0] += roi_x1
                    valid_features[:, 0, 1] += roi_y1
                elif len(valid_features.shape) == 2:  # (N, 2) 形状
                    # (N, 2) 形状の場合は直接座標を更新
                    valid_features[:, 0] += roi_x1
                    valid_features[:, 1] += roi_y1
                    # (N, 1, 2) 形状に変換
                    valid_features = valid_features.reshape(-1, 1, 2)
                
                self.get_logger().debug(f"     - {label}: ROI座標をフル座標に変換完了")
                
                # バウンディングボックスを更新（形状を考慮）
                if len(valid_features) > 0:
                    # valid_featuresは常に(N, 1, 2)形状にリシェイプ済み
                    x_coords = valid_features[:, 0, 0]
                    y_coords = valid_features[:, 0, 1]
                        
                    new_bbox = [
                        float(np.min(x_coords)),
                        float(np.min(y_coords)),
                        float(np.max(x_coords)),
                        float(np.max(y_coords))
                    ]
                    
                    # データを更新
                    self.tracked_boxes_per_label[label] = new_bbox
                    self.tracked_features_per_label[label] = valid_features
                    updated_count += 1
            
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
                else:
                    self.get_logger().warn("GroundingDINO検出結果が空です")
                    
                # 完了したタスクをクリア
                self.gdino_future = None
                    
            except Exception as e:
                self.get_logger().error(f"GroundingDINO結果統合エラー: {repr(e)}")
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
                    self.get_logger().warn("検出データにタイムスタンプがありません、通常初期化に切り替え")
                    current_gray = self._get_current_gray_image()
                    if current_gray is not None:
                        self._initialize_tracking_boxes(current_gray)
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
                    self.get_logger().warn(f"時間差が大きすぎます ({time_gap:.3f}s > {self.max_temporal_gap}s)、通常初期化")
                    current_gray = self._get_current_gray_image()
                    if current_gray is not None:
                        self._initialize_tracking_boxes(current_gray)
                    return
                
                # 検出時点のフル画像を取得
                detection_full_image = self._get_full_image_by_timestamp(detection_timestamp)
                if detection_full_image is None:
                    self.get_logger().warn("検出時点の画像が見つかりません、通常初期化")
                    current_gray = self._get_current_gray_image()
                    if current_gray is not None:
                        self._initialize_tracking_boxes(current_gray)
                    return
                
                # 検出時点の画像で追跡初期化
                self.get_logger().info(f"✓ 検出時点({detection_timestamp:.3f})の画像で追跡初期化")
                self.get_logger().info(f"  - 使用画像サイズ: {detection_full_image.shape}")
                self._initialize_tracking_boxes(detection_full_image)
                
                # キャッチアップ処理: 検出時点から現在まで高速追跡
                if time_gap > 0.05:  # 50ms以上の差がある場合のみキャッチアップ
                    self.get_logger().info(f"✓ キャッチアップ処理実行: {detection_timestamp:.3f} -> {current_time:.3f}")
                    
                    # バッファ内のフレーム数を計算
                    with self.temporal_lock:
                        relevant_frames = 0
                        for timestamp in self.image_timestamps:
                            if detection_timestamp < timestamp <= current_time:
                                relevant_frames += 1
                    
                    self.get_logger().info(f"  - キャッチアップ対象フレーム数: {relevant_frames}")
                    self._perform_temporal_catchup_optimized(detection_timestamp, current_time)
                else:
                    self.get_logger().info(f"✓ キャッチアップスキップ: 時間差が闾値以下 ({time_gap*1000:.1f}ms < 50ms)")
                
                # 時間遡り処理完了後、現在の画像をprev_grayに設定してリアルタイム追跡に備える
                current_gray = self._get_current_gray_image()
                if current_gray is not None:
                    self.prev_gray = current_gray.copy()
                    self.get_logger().info(f"✓ 時間遡り処理完了後、現在画像をprev_grayに設定")
                
                self.get_logger().info(f"=== 時間的整合性処理完了 ===")
                
        except Exception as e:
            self.get_logger().error(f"時間整合性追跡初期化エラー: {e}")
            # フォールバック: 通常の初期化
            current_gray = self._get_current_gray_image()
            if current_gray is not None:
                self._initialize_tracking_boxes(current_gray)
    
    def _perform_temporal_catchup_optimized(self, start_timestamp: float, end_timestamp: float) -> None:
        """段階的キャッチアップ処理（真の早送り処理）"""
        try:
            # キャッチアップに必要なフル画像フレームを収集
            catchup_frames = []
            with self.temporal_lock:
                for i, timestamp in enumerate(self.image_timestamps):
                    if start_timestamp < timestamp <= end_timestamp:
                        if i < len(self.full_image_buffer):
                            catchup_frames.append((timestamp, self.full_image_buffer[i].copy()))
            
            # 時間順にソート
            catchup_frames.sort(key=lambda x: x[0])
            
            if len(catchup_frames) == 0:
                self.get_logger().warn("キャッチアップ対象フレームがありません")
                return
            
            self.get_logger().info(f"段階的キャッチアップ開始: {len(catchup_frames)}フレーム")
            self.get_logger().info(f"  - 開始時刻: {start_timestamp:.3f}")
            self.get_logger().info(f"  - 終了時刻: {end_timestamp:.3f}")
            
            # 開始フレームのフル画像を取得
            start_full_image = self._get_full_image_by_timestamp(start_timestamp)
            if start_full_image is None:
                self.get_logger().warn("開始フレームの画像が見つかりません")
                return
            
            prev_gray = start_full_image  # フル画像を使用
            
            # 段階的にオプティカルフロー処理を実行
            for i, (timestamp, curr_gray) in enumerate(catchup_frames):
                self.get_logger().debug(f"  - フレーム {i+1}/{len(catchup_frames)}: {timestamp:.3f}")
                
                # 段階的オプティカルフロー更新
                self._stepwise_optical_flow_update(prev_gray, curr_gray, timestamp)
                prev_gray = curr_gray
            
            self.get_logger().info(f"段階的キャッチアップ完了: {start_timestamp:.3f} -> {end_timestamp:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"段階的キャッチアップエラー: {e}")
    
    def _stepwise_optical_flow_update(self, prev_gray: np.ndarray, curr_gray: np.ndarray, timestamp: float) -> None:
        """段階的オプティカルフロー更新（早送り処理の核心）"""
        try:
            with self.tracking_data_lock:
                updated_count = 0
                
                # 全追跡オブジェクトを段階的に更新
                for label, prev_features in list(self.tracked_features_per_label.items()):
                    if prev_features is None or len(prev_features) == 0:
                        continue
                    
                    self.get_logger().debug(f"    - {label}: {len(prev_features)}個の特徴点で段階的追跡")
                    
                    # オプティカルフロー計算
                    try:
                        curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, curr_gray, prev_features, None,
                            winSize=(self.optical_flow_win_size_x, self.optical_flow_win_size_y),
                            maxLevel=self.optical_flow_max_level,
                            criteria=self.optical_flow_criteria
                        )
                    except cv2.error as e:
                        self.get_logger().debug(f"    - {label}: オプティカルフロー計算エラー: {e}")
                        continue
                    
                    # 有効な特徴点のみを保持
                    if curr_features is not None and status is not None:
                        valid_indices = (status == 1).flatten()
                        valid_count = np.sum(valid_indices)
                        valid_ratio = valid_count / len(valid_indices) if len(valid_indices) > 0 else 0
                        
                        self.get_logger().debug(f"    - {label}: {valid_count}/{len(valid_indices)}個有効 (有効率: {valid_ratio:.2f})")
                        
                        if valid_ratio >= 0.1:  # 10%以上の特徴点が有効（緩和）
                            valid_features = curr_features[valid_indices]
                            self.tracked_features_per_label[label] = valid_features
                            
                            # バウンディングボックスを更新
                            self._update_bounding_box_from_features(label)
                            updated_count += 1
                            
                            self.get_logger().debug(f"    - {label}: {len(valid_features)}個の特徴点を更新")
                        else:
                            # 有効率が低い場合は削除
                            self.get_logger().info(f"    - {label}: 有効率が低いため削除 (有効率: {valid_ratio:.2f})")
                            self._remove_tracking_object(label)
                    else:
                        self.get_logger().info(f"    - {label}: オプティカルフロー結果が無効")
                        self._remove_tracking_object(label)
                
                self.get_logger().debug(f"    - {updated_count}個のオブジェクトを更新 ({timestamp:.3f})")
                
        except Exception as e:
            self.get_logger().error(f"段階的オプティカルフロー更新エラー: {e}")
            
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
        """高速オプティカルフロー更新（キャッチアップ用）"""
        try:
            if self.prev_gray is None or not self.tracking_valid:
                return
            
            # 簡略化されたオプティカルフロー更新
            with self.tracking_data_lock:
                for label, prev_features in list(self.tracked_features_per_label.items()):
                    if prev_features is None or len(prev_features) == 0:
                        continue
                    
                    # オプティカルフロー計算
                    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_features,
                        None,
                        winSize=(self.optical_flow_win_size_x, self.optical_flow_win_size_y),
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                    
                    # 有効な特徴点のみ保持
                    if curr_features is not None and status is not None:
                        valid_indices = (status == 1).flatten()
                        if np.any(valid_indices):
                            self.tracked_features_per_label[label] = curr_features[valid_indices]
                            
                            # バウンディングボックス更新
                            self._update_bounding_box_from_features(label)
            
            # 前フレーム更新
            self.prev_gray = gray.copy()
            
        except Exception as e:
            self.get_logger().error(f"高速オプティカルフロー更新エラー: {e}")
    
    def _get_current_gray_image(self) -> Optional[np.ndarray]:
        """現在のフル画像を取得"""
        try:
            with self.temporal_lock:
                if len(self.full_image_buffer) > 0:
                    return self.full_image_buffer[-1].copy()
            return None
        except Exception as e:
            self.get_logger().error(f"現在画像取得エラー: {e}")
            return None
    
    def _update_bounding_box_from_features(self, label: str) -> None:
        """特徴点からバウンディングボックスを更新"""
        try:
            if (label not in self.tracked_features_per_label or 
                label not in self.feature_to_center_offsets or
                label not in self.original_box_sizes):
                return
            
            features = self.tracked_features_per_label[label]
            offsets = self.feature_to_center_offsets[label]
            
            if features is None or len(features) == 0 or offsets is None or len(offsets) == 0:
                return
            
            # 相対距離を使って中心点を逆算
            predicted_centers = []
            for i, feature in enumerate(features):
                if i < len(offsets):
                    fx, fy = feature[0]
                    offset_x, offset_y = offsets[i]
                    predicted_center_x = fx - offset_x
                    predicted_center_y = fy - offset_y
                    predicted_centers.append([predicted_center_x, predicted_center_y])
            
            if len(predicted_centers) > 0:
                # 平均中心点を計算
                avg_center = np.mean(predicted_centers, axis=0)
                
                # 元のサイズでバウンディングボックスを再構築
                width, height = self.original_box_sizes[label]
                x1 = avg_center[0] - width / 2
                y1 = avg_center[1] - height / 2
                x2 = avg_center[0] + width / 2
                y2 = avg_center[1] + height / 2
                
                # バウンディングボックス更新
                self.tracked_boxes_per_label[label] = [x1, y1, x2, y2]
                
        except Exception as e:
            self.get_logger().error(f"特徴点からのバウンディングボックス更新エラー: {e}")
    
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
        """一段階のオプティカルフロー更新（補間用）"""
        try:
            with self.tracking_data_lock:
                if not self.tracked_features_per_label:
                    return
                
                for label, prev_features in list(self.tracked_features_per_label.items()):
                    if prev_features is None or len(prev_features) == 0:
                        continue
                    
                    # オプティカルフロー計算
                    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, prev_features, None,
                        winSize=(self.optical_flow_win_size_x, self.optical_flow_win_size_y),
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                    
                    # 有効な特徴点のみ保持
                    if curr_features is not None and status is not None:
                        valid_indices = (status == 1).flatten()
                        if np.any(valid_indices):
                            # 特徴点更新
                            valid_features = curr_features[valid_indices]
                            self.tracked_features_per_label[label] = valid_features
                            
                            # 相対オフセットも更新
                            if label in self.feature_to_center_offsets:
                                valid_offsets = self.feature_to_center_offsets[label][valid_indices]
                                self.feature_to_center_offsets[label] = valid_offsets
                            
                            # バウンディングボックス更新
                            self._update_bounding_box_from_features(label)
                        else:
                            # 追跡失敗時はラベルを削除
                            if label in self.tracked_features_per_label:
                                del self.tracked_features_per_label[label]
                            if label in self.tracked_boxes_per_label:
                                del self.tracked_boxes_per_label[label]
                            if label in self.feature_to_center_offsets:
                                del self.feature_to_center_offsets[label]
                
                # 前フレーム更新
                self.prev_gray = curr_gray.copy()
                self.tracking_valid = len(self.tracked_features_per_label) > 0
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー段階更新エラー: {e}")
    
    def _integrate_new_detections_with_existing_tracking(self, current_time: float) -> None:
        """既存の追跡を維持しながら新規検出結果を統合"""
        try:
            with self.detection_data_lock:
                if self.latest_detection_data is None:
                    self.get_logger().warn("新規検出データがありません")
                    return
                
                new_boxes = self.latest_detection_data.get('boxes', [])
                new_labels = self.latest_detection_data.get('labels', [])
                detection_timestamp = self.latest_detection_data.get('original_timestamp')
                
                if len(new_boxes) == 0:
                    self.get_logger().info("新規検出ボックスがありません")
                    return
            
            # 既存の追跡データを保持
            with self.tracking_data_lock:
                existing_labels = set(self.tracked_boxes_per_label.keys())
                self.get_logger().info(f"既存追跡: {len(existing_labels)}個, 新規検出: {len(new_boxes)}個")
                
                # 新規検出と既存追跡の重複チェック・統合
                integrated_count = 0
                added_count = 0
                
                for i, new_box in enumerate(new_boxes):
                    new_label = new_labels[i] if i < len(new_labels) else f'object_{i}'
                    
                    # トラッキング対象フィルタリング
                    if not self._is_tracking_target(new_label):
                        continue
                    
                    # 既存追跡との重複チェック
                    overlap_label = self._find_overlapping_existing_tracking(new_box, new_label)
                    
                    if overlap_label:
                        # 既存追跡と重複している場合は統合
                        self.get_logger().info(f"既存追跡'{overlap_label}'と新規検出'{new_label}'を統合")
                        self._merge_detection_with_existing_tracking(overlap_label, new_box, detection_timestamp)
                        integrated_count += 1
                    else:
                        # 新しいオブジェクトとして追加
                        unique_label = self._generate_unique_label(new_label, existing_labels)
                        self.get_logger().info(f"新規オブジェクト'{unique_label}'を追加")
                        self._add_new_tracking_object(unique_label, new_box, detection_timestamp, current_time)
                        existing_labels.add(unique_label)
                        added_count += 1
                
                # 検出されなくなったオブジェクトを削除
                removed_count = self._remove_undetected_objects(new_labels)
                
                self.get_logger().info(f"統合完了: {integrated_count}個統合, {added_count}個新規追加, {removed_count}個削除")
                
        except Exception as e:
            self.get_logger().error(f"検出統合エラー: {e}")
            # フォールバック: 従来の初期化方式
            current_gray = self._get_current_gray_image()
            if current_gray is not None:
                self._initialize_tracking_boxes(current_gray)
    
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
    
    def _add_new_tracking_object(self, unique_label: str, new_box: List[float], detection_timestamp: Optional[float], current_time: float) -> None:
        """新規オブジェクトを追跡に追加"""
        try:
            # 検出時点の画像を取得して特徴点を抽出
            if detection_timestamp:
                images = self._get_roi_by_timestamp(detection_timestamp)
                if images:
                    detection_gray, detection_rgb, roi_coords = images
                    
                    # 新規オブジェクトの特徴点抽出
                    feature_points = extract_harris_corners_from_bbox(
                        detection_gray,
                        new_box,
                        self.harris_max_corners,
                        self.harris_quality_level,
                        self.harris_min_distance,
                        self.harris_block_size,
                        self.harris_k,
                        self.use_harris_detector
                    )
                    
                    # 追跡データに追加
                    if feature_points is not None and len(feature_points) > 0:
                        self._add_tracking_data(unique_label, new_box, feature_points)
                        
                        # 検出時点から現在まで高速追跡で更新
                        time_gap = current_time - detection_timestamp
                        if time_gap > 0.05:
                            self._fast_forward_new_tracking(unique_label, detection_timestamp, current_time)
                        
                        self.get_logger().info(f"新規追跡'{unique_label}'を追加: {len(feature_points)}個の特徴点")
                    else:
                        # 特徴点が取得できない場合は中心点で追加
                        center_x = (new_box[0] + new_box[2]) / 2
                        center_y = (new_box[1] + new_box[3]) / 2
                        center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                        self._add_tracking_data(unique_label, new_box, center_point)
                        self.get_logger().warn(f"新規追跡'{unique_label}'を中心点で追加")
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
    
    def _add_tracking_data(self, label: str, box: List[float], features: np.ndarray) -> None:
        """追跡データを内部構造に追加"""
        try:
            # バウンディングボックス
            self.tracked_boxes_per_label[label] = box
            
            # 特徴点
            self.tracked_features_per_label[label] = features
            
            # 中心点
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            self.tracked_centers_per_label[label] = center_point
            
            # 元のサイズ
            width = box[2] - box[0]
            height = box[3] - box[1]
            self.original_box_sizes[label] = (width, height)
            
            # 相対オフセット
            offsets = []
            for feature_point in features:
                fx, fy = feature_point[0]
                offset_x = fx - center_x
                offset_y = fy - center_y
                offsets.append([offset_x, offset_y])
            self.feature_to_center_offsets[label] = np.array(offsets, dtype=np.float32)
            
            # 検出時刻を記録
            current_time = time.time()
            self.object_last_detection_time[label] = current_time
            
            # 追跡有効化
            self.tracking_valid = True
            
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
        """単一オブジェクトの追跡更新"""
        try:
            if label not in self.tracked_features_per_label:
                return
            
            prev_features = self.tracked_features_per_label[label]
            if prev_features is None or len(prev_features) == 0:
                return
            
            # オプティカルフロー計算
            curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_features, None,
                winSize=(self.optical_flow_win_size_x, self.optical_flow_win_size_y),
                maxLevel=self.optical_flow_max_level,
                criteria=self.optical_flow_criteria
            )
            
            if curr_features is not None and status is not None:
                valid_indices = (status == 1).flatten()
                if np.any(valid_indices):
                    # 有効な特徴点のみ保持
                    valid_features = curr_features[valid_indices]
                    self.tracked_features_per_label[label] = valid_features
                    
                    # 相対オフセットも更新
                    if label in self.feature_to_center_offsets:
                        valid_offsets = self.feature_to_center_offsets[label][valid_indices]
                        self.feature_to_center_offsets[label] = valid_offsets
                    
                    # バウンディングボックス更新
                    self._update_bounding_box_from_features(label)
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
        """追跡オブジェクトを削除"""
        try:
            if label in self.tracked_features_per_label:
                del self.tracked_features_per_label[label]
            if label in self.tracked_boxes_per_label:
                del self.tracked_boxes_per_label[label]
            if label in self.tracked_centers_per_label:
                del self.tracked_centers_per_label[label]
            if label in self.feature_to_center_offsets:
                del self.feature_to_center_offsets[label]
            if label in self.original_box_sizes:
                del self.original_box_sizes[label]
            if label in self.object_last_detection_time:
                del self.object_last_detection_time[label]
                
            self.get_logger().info(f"追跡オブジェクト'{label}'を削除")
            
        except Exception as e:
            self.get_logger().error(f"追跡オブジェクト削除エラー: {e}")
    
    def _cleanup_old_tracking_data(self) -> None:
        """古い・無効な追跡データの定期的クリーンアップ"""
        try:
            with self.tracking_data_lock:
                initial_count = len(self.tracked_boxes_per_label)
                if initial_count == 0:
                    return
                
                # 無効な特徴点を持つラベルを特定
                labels_to_remove = []
                for label in list(self.tracked_boxes_per_label.keys()):
                    # 特徴点が無効または空の場合
                    if (label not in self.tracked_features_per_label or 
                        self.tracked_features_per_label[label] is None or 
                        len(self.tracked_features_per_label[label]) == 0):
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
        """単一オブジェクトの追跡初期化"""
        try:
            # 特徴点抽出
            feature_points = extract_harris_corners_from_bbox(
                gray, box,
                self.harris_max_corners,
                self.harris_quality_level,
                self.harris_min_distance,
                self.harris_block_size,
                self.harris_k,
                self.use_harris_detector
            )
            
            if feature_points is not None and len(feature_points) > 0:
                self._add_tracking_data(label, box, feature_points)
                self.get_logger().info(f"単一追跡初期化'{label}': {len(feature_points)}個の特徴点")
            else:
                # 特徴点が取得できない場合は中心点で初期化
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                self._add_tracking_data(label, box, center_point)
                self.get_logger().warn(f"単一追跡初期化'{label}': 中心点で初期化")
                
        except Exception as e:
            self.get_logger().error(f"単一追跡初期化エラー: {e}")
    
    def _initialize_tracking_boxes(self, gray: np.ndarray) -> None:
        """検出結果からバウンディングボックス追跡を初期化"""
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
                self.tracked_boxes_per_label = {}
                self.tracked_centers_per_label = {}
                self.tracked_features_per_label = {}
                self.original_box_sizes = {}
                self.feature_to_center_offsets = {}
                
                # ラベルの重複をカウントするための辞書
                label_counts = {}
                
                # 並列処理の準備: バウンディングボックスとラベルの組み合わせ
                bbox_label_pairs = []
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
                    
                    bbox_label_pairs.append((bbox_list, unique_label))
                
                # 並列特徴点検出（複数ボックスがある場合のみ）
                if len(bbox_label_pairs) > 1:
                    self.get_logger().info(f"並列特徴点検出を開始: {len(bbox_label_pairs)}個のボックス")
                    feature_results = self._extract_features_parallel(gray, bbox_label_pairs)
                else:
                    # 単一ボックスの場合は同期処理
                    feature_results = self._extract_features_sequential(gray, bbox_label_pairs)
                
                # 結果を統合
                for result in feature_results:
                    unique_label = result['label']
                    bbox_list = result['bbox']
                    feature_points = result['feature_points']
                    center_point = result['center_point']
                    
                    if feature_points is not None and len(feature_points) > 0:
                        self.tracked_features_per_label[unique_label] = feature_points
                        self.tracked_centers_per_label[unique_label] = center_point
                        
                        # 特徴点と中心点の相対距離を計算して保存
                        x1, y1, x2, y2 = bbox_list
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        offsets = []
                        for feature_point in feature_points:
                            fx, fy = feature_point[0]  # (1, 2) -> (2,)
                            offset_x = fx - center_x
                            offset_y = fy - center_y
                            offsets.append([offset_x, offset_y])
                        
                        self.feature_to_center_offsets[unique_label] = np.array(offsets, dtype=np.float32)
                        
                        self.get_logger().info(f"ラベル'{unique_label}': {len(feature_points)}個の特徴点を検出、相対距離を記録")
                    else:
                        # 特徴点が見つからない場合は中心点のみ使用
                        self.tracked_features_per_label[unique_label] = center_point
                        self.tracked_centers_per_label[unique_label] = center_point
                        self.feature_to_center_offsets[unique_label] = np.array([[0.0, 0.0]], dtype=np.float32)  # オフセットなし
                        
                        self.get_logger().warn(f"ラベル'{unique_label}': 特徴点検出失敗、中心点を使用")
                    
                    # 元のバウンディングボックスサイズを保存
                    x1, y1, x2, y2 = bbox_list
                    width = x2 - x1
                    height = y2 - y1
                    self.original_box_sizes[unique_label] = (width, height)
                    
                    # バウンディングボックスを保存（ユニークキー使用）
                    self.tracked_boxes_per_label[unique_label] = bbox_list
                    
                    method_name = f"特徴点({len(self.tracked_features_per_label[unique_label])}個)"
                    self.get_logger().info(f"ラベル'{unique_label}': ボックス{bbox_list}, {method_name}, サイズ({width:.1f}×{height:.1f})")
                
                # 前フレームのグレー画像を保存
                self.prev_gray = gray.copy()
                self.tracking_valid = len(self.tracked_boxes_per_label) > 0
                
                total_boxes = len(self.tracked_boxes_per_label)
                self.get_logger().info(f"バウンディングボックス追跡初期化完了: {total_boxes}個, 有効: {self.tracking_valid}")
                
        except Exception as e:
            self.get_logger().error(f"バウンディングボックス初期化エラー: {repr(e)}")
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
            # バウンディングボックス内の複数特徴点を抽出
            feature_points = extract_harris_corners_from_bbox(
                gray,
                bbox_list,
                self.harris_max_corners,
                self.harris_quality_level,
                self.harris_min_distance,
                self.harris_block_size,
                self.harris_k,
                self.use_harris_detector
            )
            
            # バウンディングボックスの中心点を計算
            x1, y1, x2, y2 = bbox_list
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            
            return {
                'label': unique_label,
                'bbox': bbox_list,
                'feature_points': feature_points,
                'center_point': center_point
            }
            
        except Exception as e:
            self.get_logger().error(f"特徴点検出エラー (ラベル: {unique_label}): {e}")
            return None
    
    def _update_bounding_box_from_center(self, label: str, new_center: np.ndarray) -> Optional[List[float]]:
        """新しい中心点に基づいて元のサイズでバウンディングボックスを再構築"""
        try:
            if label not in self.original_box_sizes:
                return None
            
            # 元のバウンディングボックスサイズを取得
            width, height = self.original_box_sizes[label]
            
            # 新しい中心点
            new_center_x, new_center_y = new_center
            
            # 元のサイズを保持して新しい中心点の周りに再構築
            half_width = width / 2
            half_height = height / 2
            
            updated_box = [
                new_center_x - half_width,   # x1
                new_center_y - half_height,  # y1
                new_center_x + half_width,   # x2
                new_center_y + half_height   # y2
            ]
            
            return updated_box
            
        except Exception as e:
            self.get_logger().error(f"バウンディングボックス更新エラー: {repr(e)}")
            return None
    
    def _track_optical_flow(self, gray: np.ndarray) -> None:
        """シンプルなオプティカルフローによる追跡"""
        try:
            with self.tracking_data_lock:
                if not self.tracked_features_per_label or self.prev_gray is None:
                    self.prev_gray = gray.copy()
                    return
                
                for label, prev_features in list(self.tracked_features_per_label.items()):
                    if prev_features is None or len(prev_features) == 0:
                        continue
                    
                    # オプティカルフロー計算
                    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_features, None,
                        winSize=self.optical_flow_win_size,
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                    
                    if curr_features is not None and status is not None:
                        # 有効な特徴点のみを選択
                        valid_indices = (status == 1).flatten()
                        if np.any(valid_indices):
                            valid_features = curr_features[valid_indices]
                            self.tracked_features_per_label[label] = valid_features
                            
                            # 相対距離を使ってバウンディングボックスを更新
                            self._update_bounding_box_from_features(label)
                            
                            self.get_logger().debug(f"'{label}': {len(valid_features)}個の特徴点追跡成功")
                        else:
                            # 追跡失敗時は削除
                            self._remove_tracking_object(label)
                            self.get_logger().info(f"'{label}': 追跡失敗により削除")
                    else:
                        # 計算失敗時も削除
                        self._remove_tracking_object(label)
                        self.get_logger().info(f"'{label}': 計算失敗により削除")
                
                self.prev_gray = gray.copy()
                self.tracking_valid = len(self.tracked_features_per_label) > 0
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー追跡エラー: {repr(e)}")
            # オプティカルフローエラーでもtracking_validは維持（SAM2が実行できるように）
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
                    'tracked_centers_per_label': {k: v.copy() for k, v in self.tracked_centers_per_label.items()},
                    'tracked_features_per_label': {k: v.copy() for k, v in self.tracked_features_per_label.items()}
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
        """オプティカルフロー結果の可視化・配信 (バウンディングボックス表示)"""
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
                # 追跡バウンディングボックスと中心点を描画
                for label, box in tracked_boxes_per_label.items():
                    if box is not None:
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # バウンディングボックスを描画
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 中心点を描画
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)
                        
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
            tracked_centers_per_label = tracking_data['tracked_centers_per_label']
            
            if not tracked_centers_per_label:
                return
            
            feature_points_msg = FeaturePoints()
            feature_points_msg.header.stamp = self.get_clock().now().to_msg()
            feature_points_msg.header.frame_id = 'camera_frame'
            
            all_points = []
            labels = []
            
            for label, center in tracked_centers_per_label.items():
                if center is not None:
                    x, y = center.ravel()[:2]  # 中心点の座標を取得
                    point32 = Point32()
                    point32.x = float(x)
                    point32.y = float(y)
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