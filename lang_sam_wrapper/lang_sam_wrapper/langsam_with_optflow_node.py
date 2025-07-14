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

from lang_sam.utils import draw_image
from lang_sam import LangSAM
from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam_wrapper.feature_extraction import extract_harris_corners_from_bbox

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
        self.optical_flow_win_size = (
            self.get_config_param('optical_flow_win_size_x', 21),
            self.get_config_param('optical_flow_win_size_y', 21)
        )
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
            self.tracking_valid = False
            
        with self.sam_data_lock:
            self.latest_sam_masks: Optional[Dict] = None
            self.sam_updated = False
        
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
    
    def __del__(self):
        """デストラクタ: スレッドプールの適切な終了"""
        try:
            if hasattr(self, 'background_executor'):
                self.background_executor.shutdown(wait=True)
            if hasattr(self, 'gdino_executor'):
                self.gdino_executor.shutdown(wait=True)
            if hasattr(self, 'sam2_executor'):
                self.sam2_executor.shutdown(wait=True)
            if hasattr(self, 'feature_executor'):
                self.feature_executor.shutdown(wait=True)
            if hasattr(self, 'optical_flow_executor'):
                self.optical_flow_executor.shutdown(wait=True)
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
        """画像処理メインループ（最適化版）"""
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
            
            # 最適化ログ（頻度を削減）
            if self.frame_count % 100 == 0 and self.frame_count > 0:
                self.get_logger().info(f"フレーム {self.frame_count} 処理中")
            
            # 1. GroundingDINO検出（非同期版）
            self._check_grounding_dino_results()  # 完了した非同期処理の結果を確認
            should_run_gdino = self._should_run_grounding_dino(current_time)
            if should_run_gdino:
                gdino_start = time.time()
                if self.frame_count % 30 == 0:
                    self.get_logger().info("GroundingDINO検出を非同期実行")
                self._run_grounding_dino_detection_async(image_cv)
                self.last_grounding_dino_time = current_time
                self.performance_stats['gdino_times'].append(time.time() - gdino_start)
            else:
                if self.frame_count % 200 == 0:  # ログ頻度を削減
                    time_since_last = current_time - self.last_grounding_dino_time
                    self.get_logger().info(f"GroundingDINOスキップ: {time_since_last:.1f}s/{self.grounding_dino_interval}s")
            
            # 2. バウンディングボックス追跡（最適化版）
            optflow_start = time.time()
            if self.detection_updated:
                if self.frame_count % 30 == 0:
                    self.get_logger().info("追跡初期化")
                self._initialize_tracking_boxes(gray)
                self.detection_updated = False
            else:
                self._track_optical_flow(gray)
            self.performance_stats['optflow_times'].append(time.time() - optflow_start)
            
            # 3. SAM2セグメンテーション（非同期版）
            self._check_sam2_results()  # 完了した非同期処理の結果を確認
            should_run_sam2 = self._should_run_sam2(current_time)
            if should_run_sam2 and self.tracking_valid:
                sam2_start = time.time()
                if self.frame_count % 30 == 0:
                    self.get_logger().info("SAM2セグメンテーション非同期実行")
                self._run_sam2_segmentation_async(image_cv)
                self.last_sam2_time = current_time
                self.performance_stats['sam2_times'].append(time.time() - sam2_start)
            
            # 4. 結果の可視化・配信（最適化版）
            self._publish_results(image_cv)
            
            # 性能統計更新
            frame_time = time.time() - frame_start_time
            self.performance_stats['frame_times'].append(frame_time)
            self._update_performance_stats(current_time)
            
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
    
    def _should_run_grounding_dino(self, current_time: float) -> bool:
        """GroundingDINO実行判定"""
        return (
            self.frame_count == 0 or
            current_time - self.last_grounding_dino_time >= self.grounding_dino_interval or
            not self.tracking_valid
        )
    
    def _should_run_sam2(self, current_time: float) -> bool:
        """SAM2実行判定"""
        if not self.tracked_boxes_per_label:
            return False
            
        return (
            self.enable_sam2_every_frame or
            current_time - self.last_sam2_time >= self.sam2_interval
        )
    
    def _run_grounding_dino_detection_async(self, image_cv: np.ndarray) -> None:
        """GroundingDINO検出実行（非同期処理）"""
        # 既に処理中の場合はスキップ
        if self.gdino_processing or (self.gdino_future and not self.gdino_future.done()):
            return
        
        # 新しい非同期タスクを開始
        self.gdino_processing = True
        self.gdino_future = self.gdino_executor.submit(
            self._grounding_dino_worker, image_cv.copy()
        )
    
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
                return {
                    'detection_data': results[0],
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
        """GroundingDINO非同期処理の結果を確認・統合"""
        if self.gdino_future and self.gdino_future.done():
            try:
                result = self.gdino_future.result()
                
                if result is not None:
                    # 結果を内部データに保存
                    with self.detection_data_lock:
                        self.latest_detection_data = result['detection_data']
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
        """オプティカルフローによる複数特徴点追跡"""
        try:
            with self.tracking_data_lock:
                if not self.tracked_features_per_label or self.prev_gray is None:
                    return
                
                updated_features = {}
                updated_centers = {}
                updated_boxes = {}
                
                for label, prev_features in self.tracked_features_per_label.items():
                    if prev_features is None or len(prev_features) == 0:
                        continue
                    
                    # オプティカルフロー計算（複数特徴点）
                    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_features, None,
                        winSize=self.optical_flow_win_size,
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                    
                    if curr_features is not None and status is not None:
                        # 有効な特徴点のみを選択
                        valid_indices = status.flatten() == 1
                        if np.any(valid_indices):
                            valid_features = curr_features[valid_indices]
                            
                            # 有効な特徴点がある場合
                            if len(valid_features) > 0:
                                # 相対距離情報を取得
                                if label in self.feature_to_center_offsets:
                                    original_offsets = self.feature_to_center_offsets[label]
                                    valid_offsets = original_offsets[valid_indices]
                                    
                                    # 各特徴点から相対距離を使って中心点を逆算
                                    predicted_centers = []
                                    for i, feature in enumerate(valid_features):
                                        fx, fy = feature[0]  # (1, 2) -> (2,)
                                        offset_x, offset_y = valid_offsets[i]
                                        predicted_center_x = fx - offset_x
                                        predicted_center_y = fy - offset_y
                                        predicted_centers.append([predicted_center_x, predicted_center_y])
                                    
                                    # 予測された中心点の平均を最終的な中心点とする
                                    predicted_centers = np.array(predicted_centers)
                                    final_center_x = np.mean(predicted_centers[:, 0])
                                    final_center_y = np.mean(predicted_centers[:, 1])
                                    
                                    new_center = np.array([[[final_center_x, final_center_y]]], dtype=np.float32)
                                    
                                    # バウンディングボックスを更新（元のサイズを保持）
                                    updated_box = self._update_bounding_box_from_center(label, [final_center_x, final_center_y])
                                    if updated_box is not None:
                                        updated_features[label] = valid_features
                                        updated_centers[label] = new_center
                                        updated_boxes[label] = updated_box
                                        
                                        # 有効なオフセットも更新
                                        self.feature_to_center_offsets[label] = valid_offsets
                                        
                                        valid_count = len(valid_features)
                                        if label in self.original_box_sizes:
                                            width, height = self.original_box_sizes[label]
                                            self.get_logger().debug(f"ラベル'{label}': {valid_count}個の特徴点追跡成功, 中心({final_center_x:.1f}, {final_center_y:.1f}), サイズ({width:.1f}×{height:.1f})")
                                        else:
                                            self.get_logger().debug(f"ラベル'{label}': {valid_count}個の特徴点追跡成功, 中心({final_center_x:.1f}, {final_center_y:.1f})")
                                    else:
                                        self.get_logger().warn(f"ラベル'{label}': バウンディングボックス更新失敗")
                                else:
                                    self.get_logger().warn(f"ラベル'{label}': 相対距離情報が見つかりません")
                            else:
                                self.get_logger().warn(f"ラベル'{label}': 有効な特徴点なし")
                        else:
                            self.get_logger().warn(f"ラベル'{label}': 全特徴点追跡失敗")
                    else:
                        self.get_logger().warn(f"ラベル'{label}': オプティカルフロー計算失敗")
                
                self.tracked_features_per_label = updated_features
                self.tracked_centers_per_label = updated_centers
                self.tracked_boxes_per_label = updated_boxes
                self.prev_gray = gray.copy()
                self.tracking_valid = len(self.tracked_features_per_label) > 0
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー追跡エラー: {repr(e)}")
            self.tracking_valid = False
    
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
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()