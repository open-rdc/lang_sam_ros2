"""
新しいフロー: テキストプロンプト → GroundingDINO → 中心点追跡 → SAM2セグメンテーション

処理順序（同期処理でマスク位置合わせ保証）:
1. GroundingDINOでバウンディングボックス検出（同期）
2. オプティカルフローでバウンディングボックス中心点追跡（同期）
3. SAM2で追跡された中心点を入力にセグメンテーション（同期）
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
from lang_sam_wrapper.feature_extraction import (
    extract_harris_corners_from_bbox,
    select_best_feature_point,
    validate_tracking_points,
    extract_edge_centroid_from_bbox
)

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


class NewLangSamOptFlowNode(Node):
    """
    新しいフロー: GroundingDINO → 中心点追跡 → SAM2セグメンテーション
    
    同期処理でマスク位置合わせを保証:
    - GroundingDINO: バウンディングボックス検出（同期）
    - オプティカルフロー: バウンディングボックス中心点追跡（同期）
    - SAM2: 追跡された中心点でセグメンテーション（同期）
    - 可視化: 非同期処理で性能向上
    """
    
    @contextmanager
    def _gpu_memory_context(self):
        """GPU メモリ管理用コンテキストマネージャー（軽量化版）"""
        try:
            # 開始時のクリーンアップを削除（処理速度優先）
            yield
        finally:
            # 必要最小限のクリーンアップのみ
            if torch.cuda.is_available() and self.frame_count % 50 == 0:  # 50フレームに1回のみ
                torch.cuda.empty_cache()
    
    def __init__(self):
        super().__init__('new_langsam_optflow_node')
        
        # コールバックグループの設定
        self._init_callback_groups()
        
        # パラメータの初期化
        self._init_parameters()
        
        # 内部状態の初期化
        self._init_state()
        
        # ROS通信の設定
        self._init_ros_communication()
        
        self.get_logger().info("新しいフロー LangSAM + Optical Flow Node 起動完了")
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
        # 既存のパラメータ
        self.text_prompt = self.get_config_param('text_prompt')
        self.box_threshold = self.get_config_param('box_threshold')
        self.text_threshold = self.get_config_param('text_threshold')
        self.tracking_targets_str = self.get_config_param('tracking_targets')
        self.tracking_targets = self._parse_tracking_targets(self.tracking_targets_str)
        
        # 新フロー用パラメータ
        self.grounding_dino_interval = self.get_config_param('grounding_dino_interval', 3.0)
        self.sam2_interval = self.get_config_param('sam2_interval', 0.1)  # 毎フレーム実行
        self.enable_sam2_every_frame = self.get_config_param('enable_sam2_every_frame', True)
        
        # Harris corner検出パラメータ
        self.harris_max_corners = self.get_config_param('harris_max_corners', 10)
        self.harris_quality_level = self.get_config_param('harris_quality_level', 0.01)
        self.harris_min_distance = self.get_config_param('harris_min_distance', 10)
        self.harris_block_size = self.get_config_param('harris_block_size', 3)
        self.harris_k = self.get_config_param('harris_k', 0.04)
        self.use_harris_detector = self.get_config_param('use_harris_detector', True)
        
        # 特徴点選択パラメータ
        self.feature_selection_method = self.get_config_param('feature_selection_method', 'harris_response')
        
        # オプティカルフローパラメータ
        self.optical_flow_win_size = (
            self.get_config_param('optical_flow_win_size_x', 15),
            self.get_config_param('optical_flow_win_size_y', 15)
        )
        self.optical_flow_max_level = self.get_config_param('optical_flow_max_level', 2)
        self.optical_flow_criteria_eps = self.get_config_param('optical_flow_criteria_eps', 0.03)
        self.optical_flow_criteria_max_count = self.get_config_param('optical_flow_criteria_max_count', 10)
        
        # 追跡検証パラメータ
        self.max_displacement = self.get_config_param('max_displacement', 50.0)
        self.min_valid_ratio = self.get_config_param('min_valid_ratio', 0.5)
        
        # エッジ検出パラメータ
        self.canny_low_threshold = self.get_config_param('canny_low_threshold', 50)
        self.canny_high_threshold = self.get_config_param('canny_high_threshold', 150)
        self.min_edge_pixels = self.get_config_param('min_edge_pixels', 10)
        
        # SAM2パラメータ
        self.sam_model = self.get_config_param('sam_model', 'sam2.1_hiera_tiny')
        self.get_logger().info(f"DEBUG: sam_model最終値: {self.sam_model}")
        
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
            
        with self.tracking_data_lock:
            self.prev_gray: Optional[np.ndarray] = None
            self.tracked_boxes_per_label: Dict[str, List[float]] = {}  # バウンディングボックス追跡用
            self.tracked_centers_per_label: Dict[str, np.ndarray] = {}  # 中心点追跡用
            self.original_box_sizes: Dict[str, Tuple[float, float]] = {}  # 初期バウンディングボックスサイズ (width, height)
            self.tracking_valid = False
            
        with self.sam_data_lock:
            self.latest_sam_masks: Optional[Dict] = None
            self.sam_updated = False
        
        # 非同期処理用（可視化のみ）
        self.background_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Visualization")
        self.visualization_future = None
        
        # タイミング管理
        self.frame_count = 0
        self.last_grounding_dino_time = 0
        self.last_sam2_time = 0
    
    def __del__(self):
        """デストラクタ: スレッドプールの適切な終了"""
        try:
            if hasattr(self, 'background_executor'):
                self.background_executor.shutdown(wait=True)
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
            if name == 'sam_model':
                self.get_logger().info(f"DEBUG: sam_model取得開始, default_value={default_value}")
            
            if default_value is not None:
                self.declare_parameter(name, default_value)
            else:
                # デフォルト値がない場合は適切なデフォルト値を設定
                if name == 'text_prompt':
                    default_value = "white line. red pylon. human. wall. car. building. mobility. road."
                elif name == 'box_threshold':
                    default_value = 0.3
                elif name == 'text_threshold':
                    default_value = 0.3
                elif name == 'tracking_targets':
                    default_value = "white line. red pylon. human. car. mobility."
                elif name == 'sam_model':
                    default_value = "sam2.1_hiera_tiny"
                elif name.startswith('grounding_dino_'):
                    default_value = 3.0
                elif name.startswith('sam2_'):
                    default_value = 0.1 if 'interval' in name else True
                elif name.startswith('harris_'):
                    default_value = 10 if 'corners' in name else 0.01
                elif name.startswith('optical_flow_'):
                    default_value = 15 if 'size' in name else 2
                elif name.startswith('feature_selection_'):
                    default_value = "center_point"
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
            
            if name == 'sam_model':
                self.get_logger().info(f"DEBUG: sam_model取得完了, value={value}")
            
            self.get_logger().info(f"パラメータ '{name}': {value}")
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
        self.get_logger().info(f"オプティカルフロー: バウンディングボックス中心点で追跡")
        self.get_logger().info(f"SAM2入力: バウンディングボックス中心点")
        self.get_logger().info(f"追跡対象: {self.tracking_targets}")
    
    def image_callback(self, msg: ROSImage) -> None:
        """メイン画像処理コールバック"""
        with self.processing_lock:
            self._process_image(msg)
    
    def _process_image(self, msg: ROSImage) -> None:
        """画像処理メインループ（GroundingDINO・オプティカルフロー・SAM2は同期処理、可視化のみ非同期）"""
        try:
            # 画像変換
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if image_cv is None or image_cv.size == 0:
                self.get_logger().error("受信した画像が空です")
                return
                
            image_cv = image_cv.astype(np.uint8, copy=True)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
            
            current_time = time.time()
            
            # フレーム処理開始ログ
            if self.frame_count % 30 == 0:  # 30フレームごとに詳細ログ
                self.get_logger().info(f"フレーム {self.frame_count} 処理開始")
            
            # 1. GroundingDINO検出（定期実行、同期処理でマスク位置合わせ保証）
            if self._should_run_grounding_dino(current_time):
                self.get_logger().info("GroundingDINO検出を実行（同期）")
                self._run_grounding_dino_detection_sync(image_cv)
                self.last_grounding_dino_time = current_time
            
            # 2. バウンディングボックス追跡（同期処理でマスク位置合わせ保証）
            if self.detection_updated:
                self.get_logger().info("バウンディングボックス追跡を初期化")
                self._initialize_tracking_boxes(gray)
                self.detection_updated = False
            else:
                self._track_optical_flow(gray)
            
            # 3. SAM2セグメンテーション（同期処理でマスク位置合わせ保証）
            if self._should_run_sam2(current_time):
                self.get_logger().info("SAM2セグメンテーションを実行")
                self._run_sam2_segmentation(image_cv)
                self.last_sam2_time = current_time
            
            # 4. 結果の可視化・配信（非同期処理で性能向上）
            self._publish_results(image_cv)
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"画像処理エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
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
    
    def _run_grounding_dino_detection_sync(self, image_cv: np.ndarray) -> None:
        """GroundingDINO検出実行（同期処理）"""
        try:
            if self.gdino_model is None:
                self.get_logger().warn("GroundingDINOモデルが初期化されていません")
                return
                
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
            
            # 結果を内部データに保存
            with self.detection_data_lock:
                if results and len(results) > 0:
                    self.latest_detection_data = results[0]
                    self.detection_updated = True
                    
                    num_detections = len(self.latest_detection_data.get('boxes', []))
                    labels = self.latest_detection_data.get('labels', [])
                    self.get_logger().info(f"GroundingDINO検出: {num_detections}個, ラベル: {labels}")
                else:
                    self.get_logger().warn("GroundingDINO検出結果が空です")
                    
        except Exception as e:
            self.get_logger().error(f"GroundingDINO検出エラー: {repr(e)}")
    
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
                self.original_box_sizes = {}
                
                # ラベルの重複をカウントするための辞書
                label_counts = {}
                
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
                    
                    self.get_logger().info(f"処理中: ラベル'{original_label}' -> '{unique_label}', ボックス={bbox}")
                    
                    # バウンディングボックスをリストに変換
                    if hasattr(bbox, 'cpu'):  # PyTorch tensor の場合
                        bbox_list = bbox.cpu().numpy().tolist()
                    else:
                        bbox_list = list(bbox)
                    
                    # バウンディングボックスの中心点を計算
                    x1, y1, x2, y2 = bbox_list
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                    
                    # 元のバウンディングボックスサイズを保存
                    width = x2 - x1
                    height = y2 - y1
                    self.original_box_sizes[unique_label] = (width, height)
                    
                    # バウンディングボックスと中心点を保存（ユニークキー使用）
                    self.tracked_boxes_per_label[unique_label] = bbox_list
                    self.tracked_centers_per_label[unique_label] = center_point
                    
                    method_name = "中心点"
                    self.get_logger().info(f"ラベル'{unique_label}': ボックス{bbox_list}, {method_name}点({center_x:.1f}, {center_y:.1f}), サイズ({width:.1f}×{height:.1f})")
                
                # 前フレームのグレー画像を保存
                self.prev_gray = gray.copy()
                self.tracking_valid = len(self.tracked_boxes_per_label) > 0
                
                total_boxes = len(self.tracked_boxes_per_label)
                self.get_logger().info(f"バウンディングボックス追跡初期化完了: {total_boxes}個, 有効: {self.tracking_valid}")
                
        except Exception as e:
            self.get_logger().error(f"バウンディングボックス初期化エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
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
        """オプティカルフローによるバウンディングボックス中心点追跡"""
        try:
            with self.tracking_data_lock:
                if not self.tracked_centers_per_label or self.prev_gray is None:
                    return
                
                updated_centers = {}
                updated_boxes = {}
                
                for label, prev_center in self.tracked_centers_per_label.items():
                    if prev_center is None:
                        continue
                    
                    # オプティカルフロー計算
                    curr_center, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_center, None,
                        winSize=self.optical_flow_win_size,
                        maxLevel=self.optical_flow_max_level,
                        criteria=self.optical_flow_criteria
                    )
                    
                    if curr_center is not None and status is not None and status[0] == 1:
                        # 追跡された中心点を更新
                        updated_centers[label] = curr_center
                        
                        # バウンディングボックスを更新
                        updated_box = self._update_bounding_box_from_center(label, curr_center[0, 0])
                        if updated_box is not None:
                            updated_boxes[label] = updated_box
                        
                        new_x, new_y = curr_center[0, 0]
                        if label in self.original_box_sizes:
                            width, height = self.original_box_sizes[label]
                            self.get_logger().debug(f"ラベル'{label}': 中心点追跡成功 ({new_x:.1f}, {new_y:.1f}), 保持サイズ({width:.1f}×{height:.1f})")
                        else:
                            self.get_logger().debug(f"ラベル'{label}': 中心点追跡成功 ({new_x:.1f}, {new_y:.1f})")
                    else:
                        self.get_logger().warn(f"ラベル'{label}': 中心点追跡失敗")
                
                self.tracked_centers_per_label = updated_centers
                self.tracked_boxes_per_label = updated_boxes
                self.prev_gray = gray.copy()
                self.tracking_valid = len(self.tracked_centers_per_label) > 0
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー追跡エラー: {repr(e)}")
            self.tracking_valid = False
    
    def _run_sam2_segmentation_optimized(self, image_cv: np.ndarray, points_data: Dict) -> Dict:
        """最適化されたSAM2セグメンテーション実行 (ポイント入力版)"""
        try:
            if self.sam_model_instance is None:
                self.get_logger().warn("SAM2モデルが初期化されていません")
                return None
                
            all_points = points_data['points']
            all_labels = points_data['labels']
            
            if not all_points:
                return None
            
            # GPU最適化: torch.no_grad()とメモリ効率化
            with torch.no_grad():
                # SAM2モデルに画像を一度だけ設定（効率化）
                self.sam_model_instance.predictor.set_image(image_cv)
                
                # バッチ処理でSAM2推論を実行
                masks_list = []
                scores_list = []
                logits_list = []
                
                for point in all_points:
                    # 特徴点座標を取得
                    if len(point.shape) == 3:  # (1, 1, 2)形式の場合
                        point_coords = point.reshape(1, -1)  # (1, 2)に変換
                    else:
                        point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
                    
                    point_labels = np.array([1], dtype=np.int32)  # 前景点として設定
                    
                    # SAM2のpoint_coords入力で推論
                    masks, scores, logits = self.sam_model_instance.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
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
            
            # 結果を結合
            combined_masks = np.array(masks_list) if masks_list else np.array([])
            combined_scores = np.array(scores_list) if scores_list else np.array([])
            combined_logits = np.array(logits_list) if logits_list else np.array([])
            
            return {
                'masks': combined_masks,
                'scores': combined_scores,
                'logits': combined_logits,
                'labels': all_labels,
                'points': all_points
            }
                
        except Exception as e:
            self.get_logger().error(f"SAM2セグメンテーションエラー: {repr(e)}")
            return None
    
    
    def _run_sam2_segmentation(self, image_cv: np.ndarray) -> None:
        """SAM2セグメンテーション実行（同期処理、中心点入力版）"""
        try:
            with self.tracking_data_lock:
                if not self.tracked_centers_per_label:
                    return
                
                # 現在の追跡中心点データを取得
                centers_data = {
                    'points': [center for center in self.tracked_centers_per_label.values() if center is not None],
                    'labels': [label for label, center in self.tracked_centers_per_label.items() if center is not None]
                }
                
                if not centers_data['points']:
                    return
            
            # SAM2処理を同期実行（中心点入力）
            result = self._run_sam2_segmentation_optimized(image_cv, centers_data)
            
            if result:
                # 結果を保存
                with self.sam_data_lock:
                    self.latest_sam_masks = result
                    self.sam_updated = True
                    
                    self.get_logger().info(f"SAM2セグメンテーション完了: {len(result['masks'])}個のマスク（中心点入力）")
                
        except Exception as e:
            self.get_logger().error(f"SAM2セグメンテーションエラー: {repr(e)}")
    
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
                    'tracked_centers_per_label': {k: v.copy() for k, v in self.tracked_centers_per_label.items()}
                }
            
            with self.sam_data_lock:
                sam_data = dict(self.latest_sam_masks) if self.latest_sam_masks else None
            
            # 非同期で可視化処理を開始
            self.visualization_future = self.background_executor.submit(
                self._visualization_worker, image_cv.copy(), tracking_data, sam_data
            )
                
        except Exception as e:
            self.get_logger().error(f"結果配信エラー: {repr(e)}")
    
    def _visualization_worker(self, image_cv: np.ndarray, tracking_data: Dict, sam_data: Optional[Dict]) -> None:
        """可視化処理ワーカー (別スレッドで実行)"""
        try:
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
    
    def _publish_optical_flow_result_worker(self, image_cv: np.ndarray, tracking_data: Dict) -> None:
        """オプティカルフロー結果の可視化・配信 (バウンディングボックス表示)"""
        try:
            tracked_boxes_per_label = tracking_data['tracked_boxes_per_label']
            
            # デバッグログ追加
            self.get_logger().info(f"オプティカルフロー結果配信開始: tracked_boxes_per_label={len(tracked_boxes_per_label)}")
            
            # 画像をコピー
            result_image = image_cv.copy()
            
            # 追跡ボックスがない場合でも空でない画像を配信
            if not tracked_boxes_per_label:
                self.get_logger().warn("追跡ボックスがありません。元画像を配信します。")
                # 元画像にテキストを追加
                cv2.putText(result_image, "Waiting for tracking...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                # 追跡バウンディングボックスと中心点を描画
                box_count = 0
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
                        
                        box_count += 1
                
                self.get_logger().info(f"追跡ボックス描画完了: {box_count}個")
            
            # ROS メッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(result_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.optflow_image_pub.publish(ros_msg)
            
            self.get_logger().info("オプティカルフロー結果配信完了")
                
        except Exception as e:
            self.get_logger().error(f"オプティカルフロー結果配信エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
    def _publish_sam2_result_worker(self, image_cv: np.ndarray, sam_data: Dict) -> None:
        """SAM2セグメンテーション結果の可視化・配信（ワーカー版、ポイント入力版）"""
        try:
            masks = sam_data['masks']
            labels = sam_data['labels']
            scores = sam_data['scores']
            points = sam_data['points']
            
            # 中心点から仮のバウンディングボックスを生成（可視化のため）
            dummy_boxes = []
            for point in points:
                if len(point.shape) == 3:  # (1, 1, 2)形式
                    x, y = point[0, 0]
                else:
                    x, y = point[0], point[1]
                # 中心点周りに20x20ピクセルのボックスを作成
                dummy_boxes.append([x-10, y-10, x+10, y+10])
            
            # draw_imageを使用して可視化
            annotated_image = draw_image(
                image_rgb=image_cv,
                masks=np.array(masks),
                xyxy=np.array(dummy_boxes) if dummy_boxes else np.array([]),
                probs=scores,
                labels=labels
            )
            
            # ROS メッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.sam_image_pub.publish(ros_msg)
            
            self.get_logger().info("SAM2結果配信完了")
                
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
            sam_masks_msg.probs = scores.tolist()
            
            # マスクをROSImageに変換
            for i, mask in enumerate(masks):
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
                mask_msg.header = sam_masks_msg.header
                sam_masks_msg.masks.append(mask_msg)
            
            self.sam_masks_pub.publish(sam_masks_msg)
            
        except Exception as e:
            self.get_logger().error(f"SAMマスク配信エラー: {repr(e)}")
    
    def _publish_sam2_result(self, image_cv: np.ndarray) -> None:
        """SAM2セグメンテーション結果の可視化・配信"""
        try:
            with self.sam_data_lock:
                if self.latest_sam_masks is None:
                    return
                
                masks = self.latest_sam_masks['masks']
                labels = self.latest_sam_masks['labels']
                scores = self.latest_sam_masks['scores']
                points = self.latest_sam_masks['points']
                
                self.get_logger().info(f"SAM2結果詳細: マスク数={len(masks)}, ラベル数={len(labels)}, スコア数={len(scores)}")
                
                # 形状の不一致を修正
                if len(masks) != len(labels):
                    # マスクの数に合わせてラベルを調整
                    if len(masks) == 1 and len(labels) > 1:
                        # 1つのマスクに複数のラベルがある場合、最初のラベルを使用
                        labels = [labels[0]]
                    elif len(masks) > 1 and len(labels) == 1:
                        # 複数のマスクに1つのラベルがある場合、ラベルを複製
                        labels = labels * len(masks)
                
                # スコアの調整
                if len(scores) != len(masks):
                    if len(scores) == 1:
                        scores = np.array([scores[0]] * len(masks))
                    else:
                        scores = scores[:len(masks)]
                
                # 中心点から仮のバウンディングボックスを生成（可視化のため）
                dummy_boxes = []
                for point in points:
                    if len(point.shape) == 3:  # (1, 1, 2)形式
                        x, y = point[0, 0]
                    else:
                        x, y = point[0], point[1]
                    # 中心点周りに20x20ピクセルのボックスを作成
                    dummy_boxes.append([x-10, y-10, x+10, y+10])
                
                # draw_imageを使用して可視化
                annotated_image = draw_image(
                    image_rgb=image_cv,
                    masks=np.array(masks),
                    xyxy=np.array(dummy_boxes) if dummy_boxes else np.array([]),
                    probs=scores,
                    labels=labels
                )
                
                # ROS メッセージとして配信
                ros_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
                ros_msg.header.stamp = self.get_clock().now().to_msg()
                ros_msg.header.frame_id = 'camera_frame'
                self.sam_image_pub.publish(ros_msg)
                
                self.get_logger().info("SAM2結果配信完了")
                
        except Exception as e:
            self.get_logger().error(f"SAM2結果配信エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
    def _publish_feature_points(self, image_cv: np.ndarray) -> None:
        """バウンディングボックス中心点情報の配信"""
        try:
            with self.tracking_data_lock:
                if not self.tracked_centers_per_label:
                    return
                
                feature_points_msg = FeaturePoints()
                feature_points_msg.header.stamp = self.get_clock().now().to_msg()
                feature_points_msg.header.frame_id = 'camera_frame'
                
                all_points = []
                labels = []
                
                for label, center in self.tracked_centers_per_label.items():
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
    
    def _publish_sam_masks(self) -> None:
        """SAMマスクメッセージの配信"""
        try:
            with self.sam_data_lock:
                if self.latest_sam_masks is None:
                    return
                
                masks = self.latest_sam_masks['masks']
                labels = self.latest_sam_masks['labels']
                scores = self.latest_sam_masks['scores']
                
                sam_masks_msg = SamMasks()
                sam_masks_msg.header.stamp = self.get_clock().now().to_msg()
                sam_masks_msg.header.frame_id = 'camera_frame'
                
                sam_masks_msg.labels = [f"newsam_{label}" for label in labels]
                sam_masks_msg.probs = scores.tolist()
                
                # バウンディングボックスとマスクを設定
                boxes = []
                for mask in masks:
                    # マスクからバウンディングボックスを計算
                    y_indices, x_indices = np.where(mask > 0.5)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, y1 = np.min(x_indices), np.min(y_indices)
                        x2, y2 = np.max(x_indices), np.max(y_indices)
                        boxes.extend([float(x1), float(y1), float(x2), float(y2)])
                    else:
                        boxes.extend([0.0, 0.0, 10.0, 10.0])
                    
                    # マスクをROSメッセージに変換
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
                    mask_msg.header.stamp = sam_masks_msg.header.stamp
                    mask_msg.header.frame_id = sam_masks_msg.header.frame_id
                    sam_masks_msg.masks.append(mask_msg)
                
                sam_masks_msg.boxes = boxes
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
        node = NewLangSamOptFlowNode()
        
        # マルチスレッド実行器を使用
        executor = MultiThreadedExecutor(num_threads=4)
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