import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from lang_sam_msgs.msg import SamMasks
from lang_sam.utils import draw_image
from lang_sam import LangSAM
from PIL import Image
import os
import torch
import warnings
import threading



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
    """
    
    def __init__(self):
        super().__init__('optflow_node')
        
        # コールバックグループの設定
        self._init_callback_groups()
        
        # パラメータの初期化
        self._init_parameters()
        
        # 内部状態の初期化
        self._init_state()
        
        # ROS通信の設定
        self._init_ros_communication()
        
        self.get_logger().info("LangSAM + Optical Flow Node 起動完了")
        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")
    
    def _init_callback_groups(self) -> None:
        """コールバックグループの初期化"""
        # 画像処理用のコールバックグループ（排他的）
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        # パブリッシャー用のコールバックグループ（再帰可能）
        self.publisher_callback_group = ReentrantCallbackGroup()
        # スレッドセーフ用のロック
        self.processing_lock = threading.Lock()
    
    def _init_parameters(self) -> None:
        """パラメータの宣言と取得"""
        # LangSAM parameters
        self.sam_model = self.declare_and_get_param('sam_model', 'sam2.1_hiera_small')
        self.text_prompt = self.declare_and_get_param('text_prompt', 'car. wheel.')
        
        # Tracking reset parameters
        self.reset_interval = self.declare_and_get_param('reset_interval', 60)
        
        # Tracking target parameters
        tracking_targets_str = self.declare_and_get_param('tracking_targets', "")
        self.tracking_targets = self._parse_tracking_targets(tracking_targets_str)
        
        # Grid sampling parameters
        self.grid_size = (
            self.declare_and_get_param('grid_size_y', 8),
            self.declare_and_get_param('grid_size_x', 10)
        )
        self.max_per_cell = self.declare_and_get_param('max_per_cell', 20)
        
        # Feature detection parameters
        self.quality_level = self.declare_and_get_param('quality_level', 0.001)
        self.min_distance = self.declare_and_get_param('min_distance', 1)
        self.block_size = self.declare_and_get_param('block_size', 3)
        
        # Tracking visualization parameters
        self.tracking_circle_radius = self.declare_and_get_param('tracking_circle_radius', 4)
        self.tracking_circle_color = self.declare_and_get_param('tracking_circle_color', 255)
        
        
        # Optical flow parameters
        self.optical_flow_win_size = (
            self.declare_and_get_param('optical_flow_win_size_x', 15),
            self.declare_and_get_param('optical_flow_win_size_y', 15)
        )
        self.optical_flow_max_level = self.declare_and_get_param('optical_flow_max_level', 2)
        self.optical_flow_criteria_eps = self.declare_and_get_param('optical_flow_criteria_eps', 0.03)
        self.optical_flow_criteria_max_count = self.declare_and_get_param('optical_flow_criteria_max_count', 10)
        
        # Optical flow criteria object
        self.optical_flow_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.optical_flow_criteria_max_count,
            self.optical_flow_criteria_eps
        )
    
    def _init_state(self) -> None:
        """内部状態の初期化"""
        # CUDA設定とワーニング抑制
        self._configure_cuda_environment()
        
        # LangSAM model initialization
        try:
            self.model = LangSAM(sam_type=self.sam_model)
            self.get_logger().info("LangSAMモデルの初期化が完了しました")
        except Exception as e:
            self.get_logger().error(f"LangSAMモデルの初期化に失敗しました: {repr(e)}")
            # フォールバック: モデルなしで動作
            self.model = None
            
        self.bridge = CvBridge()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts_per_label: Dict[str, np.ndarray] = {}
        self.latest_sam_data: Optional[Dict] = None
        self.frame_count = 0
        self.sam_msg_count = 0  # SAMマスクメッセージの受信回数
        self.sam_updated = False  # LangSAM推論が完了したフラグ
    
    def _init_ros_communication(self) -> None:
        """ROS通信の設定"""
        # 画像サブスクライバー（専用コールバックグループ）
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10,
            callback_group=self.image_callback_group
        )
        # SAMマスクの配信（LangSAMの結果とトラッキング結果の両方）
        self.sam_masks_pub = self.create_publisher(
            SamMasks, '/sam_masks', 10,
            callback_group=self.publisher_callback_group
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

    def declare_and_get_param(self, name: str, default_value) -> any:
        """パラメータを宣言して取得するヘルパー関数
        
        Args:
            name: パラメータ名
            default_value: デフォルト値
            
        Returns:
            パラメータの値
        """
        self.declare_parameter(name, default_value)
        if isinstance(default_value, str):
            return self.get_parameter(name).get_parameter_value().string_value
        else:
            return self.get_parameter(name).value
    
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
        # スレッドセーフティのためのロック
        with self.processing_lock:
            self._process_image(msg)
    
    def _process_image(self, msg: ROSImage) -> None:
        """画像処理の実際の処理（ロック内で実行）
        
        Args:
            msg: 画像メッセージ
        """
        try:
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if image_cv is None or image_cv.size == 0:
                self.get_logger().warn("受信した画像が空です")
                return
            
            image_cv = image_cv.astype(np.uint8, copy=True)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

            # デバッグログ
            if self.frame_count % 30 == 0:  # 30フレームごとにログ出力
                self.get_logger().info(f"フレーム処理中: {self.frame_count}, 画像サイズ: {image_cv.shape}")

            is_reset = self._should_reset_tracking()
            if is_reset:
                self.get_logger().info(f"トラッキングリセット実行 (フレーム: {self.frame_count})")
                # LangSAMセグメンテーション実行
                self._run_lang_sam_segmentation(image_cv)
                # SAMの結果を使ってトラッキング初期化
                self._reset_tracking_points(gray)
            else:
                self._track_points(gray)

            # トラッキング結果を表示（draw_imageを使用）
            published = False
            if self.prev_pts_per_label:
                self._publish_tracking_result_with_draw_image(image_cv)
                published = True
            
            # デバッグログ
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"結果配信: {published}, トラッキングラベル数: {len(self.prev_pts_per_label)}")

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"image_callback エラー: {repr(e)}")

    def _should_reset_tracking(self) -> bool:
        """トラッキングの初期化が必要かを判定
        
        Returns:
            初期化が必要な場合True
        """
        # 初回またはトラッキング点がない場合は必ずリセット
        force_reset = not self.prev_pts_per_label or self.frame_count == 0
        
        # LangSAMセグメンテーションを実行する条件（ただし、処理頻度を下げる）
        should_run_sam = (
            self.frame_count % self.reset_interval == 0 or 
            self.frame_count == 0
        )
        
        # LangSAM推論が完了した場合は次のフレームでリセット
        sam_ready_for_reset = self.sam_updated
        
        reset_needed = should_run_sam or force_reset or sam_ready_for_reset
        
        if reset_needed and self.frame_count % 30 == 0:
            self.get_logger().info(f"リセット判定: frame={self.frame_count}, interval={self.reset_interval}, has_pts={bool(self.prev_pts_per_label)}, sam_updated={self.sam_updated}")
        
        return reset_needed

    def _reset_tracking_points(self, gray: np.ndarray) -> None:
        """特徴点を初期化
        
        Args:
            gray: グレースケール画像
        """
        if self.latest_sam_data is None:
            return
            
        self.prev_pts_per_label = {}
        
        # ラベルごとにマスクを統合
        label_masks = self._merge_masks_by_label()
        
        # 統合されたマスクから特徴点を抽出
        for label, combined_mask in label_masks.items():
            points = sample_features_grid(
                gray, combined_mask, self.grid_size, self.max_per_cell,
                self.quality_level, self.min_distance, self.block_size
            )
            if points is not None:
                self.prev_pts_per_label[label] = points
        
        self.prev_gray = gray
        # リセット完了後、SAMフラグをクリア
        self.sam_updated = False
        self.get_logger().info(f"特徴点初期化完了: {len(self.prev_pts_per_label)}ラベル")
    
    def _merge_masks_by_label(self) -> Dict[str, np.ndarray]:
        """ラベルごとにマスクを統合（トラッキング対象のみ）
        
        Returns:
            ラベルをキーとする統合マスクの辞書
        """
        label_masks = {}
        
        for i, mask_cv in enumerate(self.latest_sam_data['masks']):
            label = (
                self.latest_sam_data['labels'][i] 
                if i < len(self.latest_sam_data['labels']) 
                else f'object_{i}'
            )
            
            # トラッキング対象フィルタリング
            if not self._is_tracking_target(label):
                continue
            
            if label not in label_masks:
                label_masks[label] = np.zeros_like(mask_cv)
            
            # 同じラベルのマスクを統合（OR演算）
            label_masks[label] = np.maximum(label_masks[label], mask_cv)
        
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
        
        self.prev_gray = gray.copy()
    
    
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
                
                if self.frame_count % 30 == 0:
                    self.get_logger().info(f"draw_imageでトラッキング結果を配信: ラベル数={len(detection_data['labels'])}")
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

            # セグメンテーション推論（エラーハンドリング強化）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 一時的にワーニングを抑制
                
                # GPU使用量を削減するため、CPUにフォールバック
                try:
                    results = self.model.predict([image_pil], [self.text_prompt])
                except RuntimeError as cuda_error:
                    if "No available kernel" in str(cuda_error) or "CUDA" in str(cuda_error):
                        self.get_logger().warn(f"CUDA エラーが発生しました。CPUモードで再試行します: {cuda_error}")
                        # CPUモードで再試行
                        self._force_cpu_mode()
                        results = self.model.predict([image_pil], [self.text_prompt])
                    else:
                        raise cuda_error

            # SAMマスクとして内部データを更新
            self._update_sam_data_from_results(results)
            
            # セグメンテーション結果を描画・配信
            self._publish_sam_result(image_cv, results)
            
            # SAMマスクメッセージを配信
            self._publish_sam_masks(results)

        except Exception as e:
            self.get_logger().error(f"LangSAMセグメンテーションエラー: {repr(e)}")
            # エラー時はダミーデータを作成してトラッキングが継続できるようにする
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
            
            self.get_logger().info(f"SAMセグメンテーション結果を配信: ラベル数={len(labels)}")

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
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"LangSAM結果をSAMマスクとして配信: {len(sam_masks_msg.labels)}個")
            
        except Exception as e:
            self.get_logger().error(f"LangSAM→SAMマスク配信エラー: {repr(e)}")
    
    def _configure_cuda_environment(self) -> None:
        """CUDA環境の設定とワーニング抑制"""
        try:
            # CUDA最適化設定
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
            
            # CUDA アーキテクチャの設定（警告を抑制）
            if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
                os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # 一般的なアーキテクチャ
            
            # PyTorchの警告を抑制
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # CUDAが利用可能かチェック
            if torch.cuda.is_available():
                self.get_logger().info(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
                # メモリ使用量を制限
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
            
            self.get_logger().info("フォールバック用SAMデータを作成しました")
            
        except Exception as e:
            self.get_logger().error(f"フォールバックデータ作成エラー: {repr(e)}")


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