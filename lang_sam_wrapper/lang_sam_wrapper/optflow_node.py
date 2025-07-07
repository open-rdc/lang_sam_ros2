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
from PIL import Image
import os
import torch
import warnings
import threading
import time
import gc
from contextlib import contextmanager



def sample_features_grid(
    gray: np.ndarray, 
    mask: np.ndarray, 
    grid_size: Tuple[int, int] = (10, 8), 
    max_per_cell: int = 40,
    quality_level: float = 0.001,
    min_distance: int = 1,
    block_size: int = 3
) -> Optional[np.ndarray]:
    """グリッドごとに特徴点を抽出する関数
    
    Args:
        gray: グレースケール画像
        mask: マスク画像
        grid_size: グリッドサイズ (y, x)
        max_per_cell: セルあたりの最大特徴点数
        
    Returns:
        抽出された特徴点の配列、見つからない場合はNone
    """
    h, w = gray.shape
    step_y = h // grid_size[0]
    step_x = w // grid_size[1]
    all_pts = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x0, y0 = j * step_x, i * step_y
            x1, y1 = min(x0 + step_x, w), min(y0 + step_y, h)

            roi_gray = gray[y0:y1, x0:x1]
            roi_mask = mask[y0:y1, x0:x1]

            points = cv2.goodFeaturesToTrack(
                roi_gray,
                mask=roi_mask,
                maxCorners=max_per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
                blockSize=block_size
            )

            if points is not None:
                points[:, 0, 0] += x0
                points[:, 0, 1] += y0
                all_pts.extend(points)

    return np.array(all_pts) if all_pts else None


class OptFlowNode(Node):
    """LangSAM + Optical Flow処理を行うROSノード
    
    テキストプロンプトでセグメンテーションを実行し、
    オプティカルフローによる物体トラッキングを実行し、
    結果を描画して配信する。
    
    スレッドセーフ設計とメモリ効率を重視した実装。
    """
    
    # すべての設定値はconfig.yamlから読み込み（ハードコーディング排除）
    
    @contextmanager
    def _gpu_memory_context(self):
        """GPU メモリ管理用コンテキストマネージャー"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def __init__(self):
        super().__init__('optflow_node')
        
        # コールバックグループの設定
        self._init_callback_groups()
        
        # パラメータの初期化
        self._init_parameters()
        
        # パラメータ検証
        self._validate_parameters()
        
        # 内部状態の初期化
        self._init_state()
        
        # ROS通信の設定
        self._init_ros_communication()
        
        self.get_logger().info("LangSAM + Optical Flow Node 起動完了")
        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")
        
        # パラメータ読み込み確認（config.yamlの設定に基づく）
        if self.enable_parameter_logging:
            self._log_parameters()
    
    def _log_parameters(self) -> None:
        """パラメータの読み込み状況をログ出力"""
        self.get_logger().info("=== パラメータ読み込み確認 ===")
        self.get_logger().info(f"SAM Model: {self.sam_model}")
        self.get_logger().info(f"Text Prompt: {self.text_prompt}")
        self.get_logger().info(f"reset_interval: {self.reset_interval}")
        self.get_logger().info(f"tracking_targets: {self.tracking_targets}")
        self.get_logger().info(f"grid_size: {self.grid_size}")
        self.get_logger().info(f"max_per_cell: {self.max_per_cell}")
        self.get_logger().info(f"quality_level: {self.quality_level}")
        self.get_logger().info(f"min_distance: {self.min_distance}")
        self.get_logger().info(f"block_size: {self.block_size}")
        self.get_logger().info(f"optical_flow_win_size: {self.optical_flow_win_size}")
        self.get_logger().info(f"optical_flow_max_level: {self.optical_flow_max_level}")
        self.get_logger().info("--- メモリ管理パラメータ ---")
        self.get_logger().info(f"max_tracking_points: {self.max_tracking_points}")
        self.get_logger().info(f"memory_cleanup_interval: {self.memory_cleanup_interval}")
        self.get_logger().info("--- CUDA設定パラメータ ---")
        self.get_logger().info(f"cuda_arch_list: {self.cuda_arch_list}")
        self.get_logger().info(f"force_cpu_mode: {self.force_cpu_mode}")
        self.get_logger().info("--- パフォーマンス設定 ---")
        self.get_logger().info(f"enable_torch_no_grad: {self.enable_torch_no_grad}")
        self.get_logger().info(f"enable_memory_optimization: {self.enable_memory_optimization}")
        self.get_logger().info(f"enable_gpu_memory_cleanup: {self.enable_gpu_memory_cleanup}")
        self.get_logger().info("=== パラメータ確認完了 ===")
    
    def _validate_parameters(self) -> None:
        """パラメータの妥当性検証"""
        validation_errors = []
        
        # 数値パラメータの範囲チェック
        if self.reset_interval <= 0:
            validation_errors.append(f"reset_interval must be > 0 seconds, got {self.reset_interval}")
        
        if self.max_per_cell <= 0:
            validation_errors.append(f"max_per_cell must be > 0, got {self.max_per_cell}")
            
        if not (0.0 <= self.quality_level <= 1.0):
            validation_errors.append(f"quality_level must be 0.0-1.0, got {self.quality_level}")
            
        if self.min_distance < 0:
            validation_errors.append(f"min_distance must be >= 0, got {self.min_distance}")
            
        if self.block_size <= 0:
            validation_errors.append(f"block_size must be > 0, got {self.block_size}")
        
        # グリッドサイズチェック
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            validation_errors.append(f"grid_size must be > 0, got {self.grid_size}")
        
        # オプティカルフローパラメータチェック
        if self.optical_flow_max_level < 0:
            validation_errors.append(f"optical_flow_max_level must be >= 0, got {self.optical_flow_max_level}")
        
        # エラーがある場合は警告
        if validation_errors:
            self.get_logger().warn("=== パラメータ検証エラー ===")
            for error in validation_errors:
                self.get_logger().warn(f"- {error}")
            self.get_logger().warn("デフォルト値または修正値で動作を継続します")
        else:
            self.get_logger().info("パラメータ検証: すべて正常")
    
    def _init_callback_groups(self) -> None:
        """コールバックグループの初期化"""
        # 画像処理用のコールバックグループ（排他的）
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        # パブリッシャー用のコールバックグループ（再帰可能）
        self.publisher_callback_group = ReentrantCallbackGroup()
        # スレッドセーフ用のロック
        self.processing_lock = threading.RLock()
        self.sam_data_lock = threading.RLock()
        self.tracking_data_lock = threading.RLock()
    
    def _init_parameters(self) -> None:
        """パラメータの宣言と取得（config.yamlのデフォルト値を使用）"""
        # LangSAM parameters
        self.sam_model = self.get_config_param('sam_model')
        self.text_prompt = self.get_config_param('text_prompt')
        
        # Tracking reset parameters
        self.reset_interval = self.get_config_param('reset_interval')
        
        # Tracking target parameters
        tracking_targets_str = self.get_config_param('tracking_targets')
        self.tracking_targets = self._parse_tracking_targets(tracking_targets_str)
        
        # Grid sampling parameters
        self.grid_size = (
            self.get_config_param('grid_size_y'),
            self.get_config_param('grid_size_x')
        )
        self.max_per_cell = self.get_config_param('max_per_cell')
        
        # Feature detection parameters
        self.quality_level = self.get_config_param('quality_level')
        self.min_distance = self.get_config_param('min_distance')
        self.block_size = self.get_config_param('block_size')
        
        # Tracking visualization parameters
        self.tracking_circle_radius = self.get_config_param('tracking_circle_radius')
        self.tracking_circle_color = self.get_config_param('tracking_circle_color')
        
        # Optical flow parameters
        self.optical_flow_win_size = (
            self.get_config_param('optical_flow_win_size_x'),
            self.get_config_param('optical_flow_win_size_y')
        )
        self.optical_flow_max_level = self.get_config_param('optical_flow_max_level')
        self.optical_flow_criteria_eps = self.get_config_param('optical_flow_criteria_eps')
        self.optical_flow_criteria_max_count = self.get_config_param('optical_flow_criteria_max_count')
        
        # Optical flow criteria object
        self.optical_flow_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.optical_flow_criteria_max_count,
            self.optical_flow_criteria_eps
        )
        
        # メモリ管理パラメータ
        self.max_tracking_points = self.get_config_param('max_tracking_points')
        self.memory_cleanup_interval = self.get_config_param('memory_cleanup_interval')
        
        # CUDA設定パラメータ
        self.cuda_arch_list = self.get_config_param('cuda_arch_list')
        self.force_cpu_mode = self.get_config_param('force_cpu_mode')
        
        # ログ設定パラメータ
        self.enable_parameter_logging = self.get_config_param('enable_parameter_logging')
        self.enable_fps_logging = self.get_config_param('enable_fps_logging')
        self.log_level_override = self.get_config_param('log_level_override')
        
        # パフォーマンス設定パラメータ
        self.enable_torch_no_grad = self.get_config_param('enable_torch_no_grad')
        self.enable_memory_optimization = self.get_config_param('enable_memory_optimization')
        self.enable_gpu_memory_cleanup = self.get_config_param('enable_gpu_memory_cleanup')
    
    def _init_state(self) -> None:
        """内部状態の初期化"""
        # CUDA設定とワーニング抑制
        self._configure_cuda_environment()
        
        # LangSAM model initialization
        try:
            self.get_logger().info(f"LangSAMモデル初期化開始: {self.sam_model}")
            self.model = LangSAM(sam_type=self.sam_model)
            self.get_logger().info("LangSAMモデルの初期化が完了しました")
            
            # モデルの詳細情報を出力
            if hasattr(self.model, 'sam') and self.model.sam is not None:
                device = next(self.model.sam.parameters()).device if hasattr(self.model.sam, 'parameters') else 'unknown'
                self.get_logger().info(f"SAMモデルデバイス: {device}")
            
        except Exception as e:
            self.get_logger().error(f"LangSAMモデルの初期化に失敗しました: {repr(e)}")
            import traceback
            self.get_logger().error(f"初期化エラートレースバック: {traceback.format_exc()}")
            # フォールバック: モデルなしで動作
            self.model = None
            
        self.bridge = CvBridge()
        
        # スレッドセーフな状態管理
        with self.tracking_data_lock:
            self.prev_gray: Optional[np.ndarray] = None
            self.prev_pts_per_label: Dict[str, np.ndarray] = {}
            
        with self.sam_data_lock:
            self.latest_sam_data: Optional[Dict] = None
            self.sam_updated = False  # LangSAM推論が完了したフラグ
            
        self.frame_count = 0
        self.sam_msg_count = 0  # SAMマスクメッセージの受信回数
        self.memory_cleanup_counter = 0  # メモリクリーンアップカウンター
        
        # 時間ベースリセット用の時刻管理
        self.last_reset_time = time.time()
    
    def _init_ros_communication(self) -> None:
        """ROS通信の設定"""
        # 画像サブスクライバー（専用コールバックグループ）
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10,
            callback_group=self.image_callback_group
        )
        # セグメンテーション結果画像の配信
        self.sam_image_pub = self.create_publisher(
            ROSImage, '/image_sam', 10,
            callback_group=self.publisher_callback_group
        )
        # オプティカルフロー結果の配信
        self.pub = self.create_publisher(
            ROSImage, '/image_optflow', 10,
            callback_group=self.publisher_callback_group
        )
        # FeaturePointsメッセージの配信
        self.feature_points_pub = self.create_publisher(
            FeaturePoints, '/image_optflow_features', 10,
            callback_group=self.publisher_callback_group
        )
        # SAMマスクメッセージの配信
        self.sam_masks_pub = self.create_publisher(
            SamMasks, '/sam_masks', 10,
            callback_group=self.publisher_callback_group
        )

    def get_config_param(self, name: str):
        """config.yamlからパラメータを取得（デフォルト値なし）
        
        Args:
            name: パラメータ名
            
        Returns:
            パラメータの値
            
        Raises:
            ValueError: パラメータが見つからない場合
        """
        try:
            # まずパラメータを宣言（config.yamlから自動取得）
            self.declare_parameter(name)
            param = self.get_parameter(name)
            value = param.value
            self.get_logger().info(f"パラメータ'{name}': {value} (config.yamlから読み込み)")
            return value
        except Exception as e:
            # パラメータが存在しない場合はエラー
            self.get_logger().error(f"パラメータ'{name}'がconfig.yamlに定義されていません: {repr(e)}")
            raise ValueError(f"Required parameter '{name}' not found in config.yaml")
    
    def declare_and_get_param(self, name: str, default_value) -> any:
        """パラメータを宣言して取得するヘルパー関数（下位互換用）
        
        Args:
            name: パラメータ名
            default_value: デフォルト値
            
        Returns:
            パラメータの値
        """
        self.declare_parameter(name, default_value)
        if isinstance(default_value, str):
            actual_value = self.get_parameter(name).get_parameter_value().string_value
        else:
            actual_value = self.get_parameter(name).value
        
        # デバッグ: パラメータ値の読み込み確認
        if actual_value != default_value:
            self.get_logger().info(f"パラメータ'{name}': {default_value} → {actual_value} (config.yamlから読み込み)")
        else:
            self.get_logger().warn(f"パラメータ'{name}': {actual_value} (デフォルト値使用 - config.yaml未反映)")
            
        return actual_value
    
    def _parse_tracking_targets(self, targets_str: str) -> List[str]:
        """トラッキング対象文字列をパース
        
        Args:
            targets_str: "white line. human. red pylon."のような文字列
            
        Returns:
            ターゲットラベルのリスト
        """
        if not targets_str.strip():
            return []
            
        # ドット+スペースで分割し、空文字列を除外
        targets = [target.strip() for target in targets_str.split('.') if target.strip()]
        self.get_logger().info(f"トラッキング対象: {targets}")
        return targets

    def image_callback(self, msg: ROSImage) -> None:
        """メイン画像処理：LangSAMセグメンテーション + 特徴点の初期化・追跡・マスク生成
        
        Args:
            msg: 画像メッセージ
        """
        # 最初の数フレームでデバッグ情報を出力
        if self.frame_count < 5:
            self.get_logger().info(f"画像受信: フレーム{self.frame_count}, サイズ={msg.width}x{msg.height}")
        
        # スレッドセーフティのためのロック
        with self.processing_lock:
            self._process_image(msg)
    
    def _process_image(self, msg: ROSImage) -> None:
        """画像処理の実際の処理（ロック内で実行）
        
        Args:
            msg: 画像メッセージ
        """
        try:
            # 画像変換
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if image_cv is None or image_cv.size == 0:
                self.get_logger().warn("受信した画像が空です")
                return
            
            image_cv = image_cv.astype(np.uint8, copy=True)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

            # トラッキングリセット判定
            is_reset = self._should_reset_tracking()
            
            if is_reset:
                # LangSAMセグメンテーション実行（メモリ管理付き）
                with self._gpu_memory_context():
                    self._run_lang_sam_segmentation(image_cv)
                    
                # SAMの結果を使ってトラッキング初期化
                with self.tracking_data_lock:
                    self._reset_tracking_points(gray)
            else:
                # 通常のトラッキング
                with self.tracking_data_lock:
                    self._track_points(gray)

            # トラッキング結果を表示
            with self.tracking_data_lock:
                if self.prev_pts_per_label:
                    self._publish_tracking_result_with_draw_image(image_cv)
                    self._publish_tracking_sam_masks(image_cv.shape[:2])
                    self._publish_feature_points(image_cv)
                else:
                    # デバッグ: トラッキング点がない場合の情報を出力
                    if self.frame_count % 30 == 0:  # 30フレームごとに警告
                        self.get_logger().warn(f"フレーム{self.frame_count}: トラッキング点がありません。SAMセグメンテーション結果待ち...")
                    
                    # 空の画像を配信（黒画像）
                    empty_image = np.zeros_like(image_cv)
                    ros_msg = self.bridge.cv2_to_imgmsg(empty_image, encoding='rgb8')
                    ros_msg.header.stamp = self.get_clock().now().to_msg()
                    ros_msg.header.frame_id = 'camera_frame'
                    self.pub.publish(ros_msg)

            # フレームカウンター更新とメモリクリーンアップ
            self.frame_count += 1
            self.memory_cleanup_counter += 1
            
            # 定期的なメモリクリーンアップ（config.yamlの設定に基づく）
            if self.enable_memory_optimization and self.memory_cleanup_counter >= self.memory_cleanup_interval:
                self._cleanup_memory()
                self.memory_cleanup_counter = 0

        except Exception as e:
            self.get_logger().error(f"image_callback エラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
    
    def _cleanup_memory(self) -> None:
        """定期的なメモリクリーンアップ"""
        try:
            # Python ガベージコレクション
            gc.collect()
            
            # トラッキング点数制限（config.yamlの設定に基づく）
            with self.tracking_data_lock:
                total_points = sum(len(pts) for pts in self.prev_pts_per_label.values())
                if total_points > self.max_tracking_points:
                    self.get_logger().info(f"トラッキング点数が上限({self.max_tracking_points})を超過: {total_points}点")
                    # 各ラベルの点数を制限
                    for label in self.prev_pts_per_label:
                        if len(self.prev_pts_per_label[label]) > self.max_tracking_points // len(self.prev_pts_per_label):
                            # 古い点を削除
                            max_points_per_label = self.max_tracking_points // len(self.prev_pts_per_label)
                            self.prev_pts_per_label[label] = self.prev_pts_per_label[label][:max_points_per_label]
            
            # GPUメモリクリーンアップ（config.yamlの設定に基づく）
            if self.enable_gpu_memory_cleanup and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.get_logger().warn(f"メモリクリーンアップエラー: {repr(e)}")

    def _should_reset_tracking(self) -> bool:
        """トラッキングの初期化が必要かを判定
        
        Returns:
            初期化が必要な場合True
        """
        current_time = time.time()
        
        # 初回またはトラッキング点がない場合は必ずリセット
        force_reset = not self.prev_pts_per_label or self.frame_count == 0
        
        # 時間ベースのリセット判定（reset_intervalに基づく）
        time_elapsed = current_time - self.last_reset_time
        should_run_sam = (
            time_elapsed >= self.reset_interval or 
            self.frame_count == 0
        )
        
        # LangSAM推論が完了した場合は次のフレームでリセット（ただし頻度制限）
        sam_ready_for_reset = self.sam_updated
        
        reset_needed = should_run_sam or force_reset or sam_ready_for_reset
        
        # デバッグログ（reset_interval動作確認）
        if should_run_sam and self.frame_count > 0:
            self.get_logger().info(f"時間ベースreset_interval動作: 経過時間={time_elapsed:.2f}秒, reset_interval={self.reset_interval}秒")
        
        # リセット実行時に時刻を更新
        if reset_needed and should_run_sam:
            self.last_reset_time = current_time
        
        if reset_needed:
            reset_reason = []
            if force_reset:
                reset_reason.append("force_reset")
            if should_run_sam:
                reset_reason.append("time_based_reset")
            if sam_ready_for_reset:
                reset_reason.append("sam_ready_for_reset")
            self.get_logger().debug(f"トラッキングリセット実行: 理由={', '.join(reset_reason)}")
        
        return reset_needed

    def _reset_tracking_points(self, gray: np.ndarray) -> None:
        """特徴点を初期化
        
        Args:
            gray: グレースケール画像
        """
        if self.latest_sam_data is None:
            self.get_logger().warn("latest_sam_dataがNullのため、トラッキングポイント初期化をスキップ")
            return
            
        self.prev_pts_per_label = {}
        
        # ラベルごとにマスクを統合
        label_masks = self._merge_masks_by_label()
        self.get_logger().info(f"統合されたラベルマスク数: {len(label_masks)}")
        
        total_points = 0
        # 統合されたマスクから特徴点を抽出
        for label, combined_mask in label_masks.items():
            points = sample_features_grid(
                gray, combined_mask, self.grid_size, self.max_per_cell,
                self.quality_level, self.min_distance, self.block_size
            )
            if points is not None:
                self.prev_pts_per_label[label] = points
                total_points += len(points)
                self.get_logger().info(f"ラベル'{label}': {len(points)}個の特徴点を抽出")
            else:
                self.get_logger().warn(f"ラベル'{label}': 特徴点が抽出できませんでした")
        
        self.get_logger().info(f"トラッキングポイント初期化完了: 合計{total_points}点, {len(self.prev_pts_per_label)}ラベル")
        
        # メモリ効率化: 必要な場合のみコピー
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
        else:
            self.prev_gray[:] = gray
            
        # リセット完了後、SAMフラグをクリア
        self.sam_updated = False
    
    def _merge_masks_by_label(self) -> Dict[str, np.ndarray]:
        """ラベルごとにマスクを統合（トラッキング対象のみ）
        
        Returns:
            ラベルをキーとする統合マスクの辞書
        """
        label_masks = {}
        processed_labels = []
        
        self.get_logger().info(f"SAMデータ処理開始: {len(self.latest_sam_data['masks'])}個のマスク")
        
        for i, mask_cv in enumerate(self.latest_sam_data['masks']):
            label = (
                self.latest_sam_data['labels'][i] 
                if i < len(self.latest_sam_data['labels']) 
                else f'object_{i}'
            )
            
            processed_labels.append(label)
            
            # トラッキング対象フィルタリング
            is_target = self._is_tracking_target(label)
            self.get_logger().debug(f"ラベル'{label}': トラッキング対象={is_target}")
            
            if not is_target:
                continue
            
            if label not in label_masks:
                label_masks[label] = np.zeros_like(mask_cv)
            
            # 同じラベルのマスクを統合（OR演算）
            label_masks[label] = np.maximum(label_masks[label], mask_cv)
        
        self.get_logger().info(f"検出ラベル: {processed_labels}")
        self.get_logger().info(f"トラッキング対象: {list(label_masks.keys())}")
        return label_masks
    
    def _is_tracking_target(self, label: str) -> bool:
        """ラベルがトラッキング対象かを判定
        
        Args:
            label: 判定するラベル
            
        Returns:
            トラッキング対象の場合True
        """
        # トラッキング対象が指定されていない場合は全てを対象とする
        if not self.tracking_targets:
            return True
            
        # 完全一致またはトラッキング対象に含まれるかをチェック
        return any(target.lower() in label.lower() or label.lower() in target.lower() 
                  for target in self.tracking_targets)

    def _track_points(self, gray: np.ndarray) -> None:
        """Optical Flowによる特徴点の追跡
        
        Args:
            gray: グレースケール画像
        """
        if not self.prev_pts_per_label or self.prev_gray is None:
            return
            
        for label in list(self.prev_pts_per_label.keys()):
            prev_pts = self.prev_pts_per_label[label]
            if prev_pts is None:
                continue
                
            # オプティカルフロー計算
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None,
                winSize=self.optical_flow_win_size,
                maxLevel=self.optical_flow_max_level,
                criteria=self.optical_flow_criteria
            )

            if new_pts is not None and status is not None:
                # 有効な特徴点のみを保持
                valid_pts = new_pts[status == 1].reshape(-1, 1, 2)
                self.prev_pts_per_label[label] = valid_pts
        
        # メモリ効率化: 必要な場合のみコピー
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
        else:
            self.prev_gray[:] = gray
    
    
    def _publish_tracking_result_with_draw_image(self, image_cv: np.ndarray) -> None:
        """トラッキング結果をdraw_imageを使って描画・配信
        
        Args:
            image_cv: 入力画像
        """
        try:
            # トラッキング点からマスクとバウンディングボックスを生成
            detection_data = self._prepare_tracking_detection_data(image_cv.shape[:2])
            
            if detection_data['masks']:
                # draw_imageを使って描画
                annotated_img = draw_image(
                    image_cv,
                    np.array(detection_data['masks']),
                    np.array(detection_data['boxes']),
                    np.array(detection_data['probs']),
                    detection_data['labels']
                )
                
                # ROSメッセージとして配信
                ros_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='rgb8')
                ros_msg.header.stamp = self.get_clock().now().to_msg()
                ros_msg.header.frame_id = 'camera_frame'
                self.pub.publish(ros_msg)
                
            else:
                self.get_logger().debug("トラッキングデータが空のため配信をスキップ")
                
        except Exception as e:
            self.get_logger().error(f"draw_imageトラッキング配信エラー: {repr(e)}")
    
    def _prepare_tracking_detection_data(self, image_shape: Tuple[int, int]) -> Dict:
        """トラッキング点から描画用データを準備
        
        Args:
            image_shape: 画像の形状 (height, width)
            
        Returns:
            描画用データの辞書
        """
        masks = []
        labels = []
        boxes = []
        probs = []
        
        for label, pts in self.prev_pts_per_label.items():
            if pts is not None and len(pts) > 0:
                # トラッキング点から小さなマスクを作成
                mask = self._create_tracking_point_mask(pts, image_shape)
                
                # バウンディングボックスを計算
                bbox = self._calculate_bounding_box_from_points(pts, image_shape)
                
                masks.append(mask.astype(np.float32))
                labels.append(f'{label}_tracking')
                boxes.append(bbox)
                probs.append(1.0)
        
        return {
            'masks': masks,
            'labels': labels,
            'boxes': boxes,
            'probs': probs
        }
    
    def _create_tracking_point_mask(self, pts: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """トラッキング点から小さなマスクを作成
        
        Args:
            pts: トラッキング点
            image_shape: 画像の形状
            
        Returns:
            マスク
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for pt in pts:
            x, y = pt.ravel()
            cv2.circle(mask, (int(x), int(y)), self.tracking_circle_radius, 255, -1)
        
        return mask
    
    def _calculate_bounding_box_from_points(self, pts: np.ndarray, image_shape: Tuple[int, int]) -> List[float]:
        """トラッキング点からバウンディングボックスを計算
        
        Args:
            pts: トラッキング点
            image_shape: 画像の形状
            
        Returns:
            バウンディングボックス [x1, y1, x2, y2]
        """
        if len(pts) == 0:
            return [0, 0, 0, 0]
        
        # 点の座標を取得
        x_coords = [pt[0][0] for pt in pts]
        y_coords = [pt[0][1] for pt in pts]
        
        # 余裕を持たせてバウンディングボックスを計算
        margin = self.tracking_circle_radius * 2
        x1 = max(0, min(x_coords) - margin)
        y1 = max(0, min(y_coords) - margin)
        x2 = min(image_shape[1], max(x_coords) + margin)
        y2 = min(image_shape[0], max(y_coords) + margin)
        
        return [float(x1), float(y1), float(x2), float(y2)]
    
    def _run_lang_sam_segmentation(self, image_cv: np.ndarray) -> None:
        """LangSAMでセグメンテーションを実行し、SAMマスクを生成・配信
        
        Args:
            image_cv: 入力画像
        """
        try:
            # モデルが初期化されていない場合はスキップ
            if self.model is None:
                self.get_logger().warn("LangSAMモデルが初期化されていません。セグメンテーションをスキップします。")
                return
                
            # OpenCV → PIL形式へ変換
            image_pil = Image.fromarray(image_cv, mode='RGB')

            # セグメンテーション推論（メモリ効率化）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 一時的にワーニングを抑制
                
                # PyTorchのno_gradでメモリ使用量を削減（config.yamlの設定に基づく）
                def _run_prediction():
                    try:
                        return self.model.predict([image_pil], [self.text_prompt])
                    except RuntimeError as cuda_error:
                        if "No available kernel" in str(cuda_error) or "CUDA" in str(cuda_error):
                            self.get_logger().warn(f"CUDA エラーが発生しました。CPUモードで再試行します: {cuda_error}")
                            # CPUモードで再試行
                            self._force_cpu_mode()
                            return self.model.predict([image_pil], [self.text_prompt])
                        else:
                            raise cuda_error
                
                if self.enable_torch_no_grad:
                    with torch.no_grad():
                        results = _run_prediction()
                else:
                    results = _run_prediction()
                
                # デバッグ: LangSAM結果を確認
                if results and len(results) > 0:
                    first_result = results[0]
                    num_masks = len(first_result.get('masks', []))
                    num_labels = len(first_result.get('labels', []))
                    self.get_logger().info(f"LangSAM結果: {num_masks}個のマスク, {num_labels}個のラベル")
                    if num_labels > 0:
                        self.get_logger().info(f"検出ラベル: {first_result.get('labels', [])}")
                else:
                    self.get_logger().warn("LangSAMセグメンテーション結果が空です")

            # SAMマスクとして内部データを更新（スレッドセーフ）
            with self.sam_data_lock:
                self._update_sam_data_from_results(results)
            
            # セグメンテーション結果を描画・配信
            self._publish_sam_result(image_cv, results)
            
            # SAMマスクメッセージを配信
            self._publish_sam_masks(results)

        except Exception as e:
            self.get_logger().error(f"LangSAMセグメンテーションエラー: {repr(e)}")
            import traceback
            self.get_logger().error(f"トレースバック: {traceback.format_exc()}")
            # エラー時はダミーデータを作成してトラッキングが継続できるようにする
            with self.sam_data_lock:
                self._create_fallback_sam_data(image_cv.shape[:2])
    
    def _update_sam_data_from_results(self, results: List[Dict]) -> None:
        """LangSAMの結果から内部SAMデータを更新
        
        Args:
            results: LangSAMの推論結果
        """
        try:
            if not results or len(results) == 0:
                self.get_logger().warn("LangSAMの結果が空です")
                return
                
            first_result = results[0]
            if 'masks' not in first_result or len(first_result['masks']) == 0:
                self.get_logger().warn("LangSAMの結果にマスクが含まれていません")
                return
            
            # マスクをuint8形式に変換
            masks_uint8 = []
            for mask in first_result['masks']:
                mask_uint8 = (mask * 255).astype(np.uint8)
                masks_uint8.append(mask_uint8)
            
            self.latest_sam_data = {
                'labels': first_result['labels'],
                'masks': masks_uint8
            }
            
            # SAMマスクメッセージの受信回数をカウント（セグメンテーション実行回数）
            self.sam_msg_count += 1
            # LangSAM推論完了フラグをセット
            self.sam_updated = True
            
        except Exception as e:
            self.get_logger().error(f"SAMデータ更新エラー: {repr(e)}")
    
    def _publish_sam_result(self, image_cv: np.ndarray, results: List[Dict]) -> None:
        """LangSAMのセグメンテーション結果を描画・配信
        
        Args:
            image_cv: 入力画像
            results: LangSAMの推論結果
        """
        try:
            if not results or len(results) == 0:
                return
                
            first_result = results[0]
            if 'masks' not in first_result or len(first_result['masks']) == 0:
                return
                
            masks = np.array(first_result['masks'])
            boxes = np.array(first_result['boxes'])
            probs = np.array(first_result.get('probs', [1.0] * len(first_result['masks'])))
            labels = first_result['labels']
            
            annotated_image = draw_image(
                image_rgb=image_cv,
                masks=masks,
                xyxy=boxes,
                probs=probs,
                labels=labels
            )

            # セグメンテーション結果の配信
            sam_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            sam_msg.header.stamp = self.get_clock().now().to_msg()
            sam_msg.header.frame_id = 'camera_frame'
            self.sam_image_pub.publish(sam_msg)
            

        except Exception as e:
            self.get_logger().error(f"SAM結果描画・配信エラー: {repr(e)}")
    
    def _publish_sam_masks(self, results: List[Dict]) -> None:
        """SAMマスクメッセージを配信
        
        Args:
            results: LangSAMの推論結果
        """
        try:
            if not results or len(results) == 0:
                return
                
            first_result = results[0]
            if 'masks' not in first_result or len(first_result['masks']) == 0:
                return
                
            sam_masks_msg = SamMasks()
            sam_masks_msg.header.stamp = self.get_clock().now().to_msg()
            sam_masks_msg.header.frame_id = 'camera_frame'
            
            masks = np.array(first_result['masks'])
            boxes = np.array(first_result['boxes'])
            probs = np.array(first_result.get('probs', [1.0] * len(first_result['masks'])))
            labels = first_result['labels']
            
            # LangSAMの結果にプレフィックスを追加
            sam_masks_msg.labels = [f"langsam_{label}" for label in labels]
            sam_masks_msg.boxes = boxes.flatten().tolist()
            sam_masks_msg.probs = probs.tolist()
            
            for mask in masks:
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
                mask_msg.header.stamp = sam_masks_msg.header.stamp
                mask_msg.header.frame_id = sam_masks_msg.header.frame_id
                sam_masks_msg.masks.append(mask_msg)
            
            self.sam_masks_pub.publish(sam_masks_msg)
            
            
        except Exception as e:
            self.get_logger().error(f"LangSAM→SAMマスク配信エラー: {repr(e)}")
    
    def _configure_cuda_environment(self) -> None:
        """CUDA環境の設定とワーニング抑制（config.yamlから設定読み込み）"""
        try:
            # CUDA最適化設定
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
            
            # CUDA アーキテクチャの設定（config.yamlから読み込み）
            if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
                os.environ['TORCH_CUDA_ARCH_LIST'] = self.cuda_arch_list
                self.get_logger().info(f"CUDA_ARCH_LIST設定: {self.cuda_arch_list}")
            
            # PyTorchの警告を抑制
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # CPUモード強制設定のチェック
            if self.force_cpu_mode:
                self.get_logger().info("config.yamlによりCPUモードが強制されています")
                return
            
            # CUDAが利用可能かチェック
            if torch.cuda.is_available() and not self.force_cpu_mode:
                self.get_logger().info(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
                # GPUメモリクリーンアップが有効な場合のみ実行
                if self.enable_gpu_memory_cleanup:
                    torch.cuda.empty_cache()
            else:
                self.get_logger().info("CUDAが利用できません。CPUモードで実行します。")
                
        except Exception as e:
            self.get_logger().warn(f"CUDA設定エラー: {repr(e)}")
    
    def _force_cpu_mode(self) -> None:
        """CPUモードに強制的に切り替える"""
        try:
            # Torchデバイスを CPU に設定
            if hasattr(self.model, 'sam'):
                if hasattr(self.model.sam, 'to'):
                    self.model.sam.to('cpu')
            
            # キャッシュをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.get_logger().info("CPUモードに切り替えました")
            
        except Exception as e:
            self.get_logger().error(f"CPUモード切り替えエラー: {repr(e)}")
    
    def _create_fallback_sam_data(self, image_shape: Tuple[int, int]) -> None:
        """エラー時のフォールバック用SAMデータを作成
        
        Args:
            image_shape: 画像の形状 (height, width)
        """
        try:
            # ダミーマスクを作成（画像全体を対象とする）
            height, width = image_shape
            dummy_mask = np.ones((height, width), dtype=np.uint8) * 255
            
            self.latest_sam_data = {
                'labels': ['fallback_object'],
                'masks': [dummy_mask]
            }
            
            # カウンターを更新
            self.sam_msg_count += 1
            
            
        except Exception as e:
            self.get_logger().error(f"フォールバックデータ作成エラー: {repr(e)}")
    
    def _publish_tracking_sam_masks(self, image_shape: Tuple[int, int]) -> None:
        """トラッキング結果をSAMマスクとして配信
        
        Args:
            image_shape: 画像の形状 (height, width)
        """
        try:
            if not self.prev_pts_per_label:
                return
                
            sam_masks_msg = SamMasks()
            sam_masks_msg.header.stamp = self.get_clock().now().to_msg()
            sam_masks_msg.header.frame_id = 'camera_frame'
            
            masks = []
            labels = []
            boxes = []
            probs = []
            
            for label, pts in self.prev_pts_per_label.items():
                if pts is not None and len(pts) > 0:
                    # トラッキング点から小さなマスクを作成
                    mask = self._create_tracking_point_mask(pts, image_shape)
                    
                    # バウンディングボックスを計算
                    bbox = self._calculate_bounding_box_from_points(pts, image_shape)
                    
                    # マスクをROSメッセージに変換
                    mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                    mask_msg.header.stamp = sam_masks_msg.header.stamp
                    mask_msg.header.frame_id = sam_masks_msg.header.frame_id
                    
                    masks.append(mask_msg)
                    labels.append(f'{label}_tracking')
                    boxes.extend(bbox)  # x1, y1, x2, y2を追加
                    probs.append(1.0)
            
            if len(masks) > 0:
                sam_masks_msg.masks = masks
                sam_masks_msg.labels = labels
                sam_masks_msg.boxes = boxes
                sam_masks_msg.probs = probs
                
                self.sam_masks_pub.publish(sam_masks_msg)
                
            
        except Exception as e:
            self.get_logger().error(f"トラッキング→SAMマスク配信エラー: {repr(e)}")

    def _publish_feature_points(self, image_cv: np.ndarray) -> None:
        """トラッキング結果をFeaturePointsメッセージとして配信
        
        Args:
            image_cv: 入力画像
        """
        try:
            if not self.prev_pts_per_label:
                return
                
            feature_points_msg = FeaturePoints()
            feature_points_msg.header.stamp = self.get_clock().now().to_msg()
            feature_points_msg.header.frame_id = 'camera_frame'
            
            # 元画像を設定
            feature_points_msg.source_image = self.bridge.cv2_to_imgmsg(image_cv, encoding='rgb8')
            feature_points_msg.source_image.header = feature_points_msg.header
            
            all_points = []
            labels = []
            point_counts = []
            boxes = []
            probs = []
            
            for label, pts in self.prev_pts_per_label.items():
                if pts is not None and len(pts) > 0:
                    # 特徴点をPoint32の配列に変換
                    label_points = []
                    for pt in pts:
                        x, y = pt.ravel()
                        point32 = Point32()
                        point32.x = float(x)
                        point32.y = float(y)
                        point32.z = 0.0
                        label_points.append(point32)
                    
                    # バウンディングボックスを計算
                    bbox = self._calculate_bounding_box_from_points(pts, image_cv.shape[:2])
                    
                    all_points.extend(label_points)
                    labels.append(f'{label}_tracking')
                    point_counts.append(len(label_points))
                    boxes.extend(bbox)  # x1, y1, x2, y2を追加
                    probs.append(1.0)
            
            if len(all_points) > 0:
                feature_points_msg.points = all_points
                feature_points_msg.labels = labels
                feature_points_msg.point_counts = point_counts
                feature_points_msg.boxes = boxes
                feature_points_msg.probs = probs
                
                self.feature_points_pub.publish(feature_points_msg)
                
            
        except Exception as e:
            self.get_logger().error(f"FeaturePoints配信エラー: {repr(e)}")


def main(args=None):
    """メイン関数"""
    rclpy.init(args=args)
    
    try:
        node = OptFlowNode()
        
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