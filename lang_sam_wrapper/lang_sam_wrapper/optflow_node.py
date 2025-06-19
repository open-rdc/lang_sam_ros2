import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from lang_sam_msgs.msg import SamMasks
from lang_sam.utils import draw_image


# Default constants (used as fallbacks if config is not available)
DEFAULT_RESET_INTERVAL = 60
DEFAULT_GRID_SIZE_Y = 8
DEFAULT_GRID_SIZE_X = 10
DEFAULT_MAX_PER_CELL = 20
DEFAULT_QUALITY_LEVEL = 0.001
DEFAULT_MIN_DISTANCE = 1
DEFAULT_BLOCK_SIZE = 3
DEFAULT_TRACKING_CIRCLE_RADIUS = 4
DEFAULT_TRACKING_CIRCLE_COLOR = 255
DEFAULT_OPTICAL_FLOW_WIN_SIZE_X = 15
DEFAULT_OPTICAL_FLOW_WIN_SIZE_Y = 15
DEFAULT_OPTICAL_FLOW_MAX_LEVEL = 2
DEFAULT_OPTICAL_FLOW_CRITERIA_EPS = 0.03
DEFAULT_OPTICAL_FLOW_CRITERIA_MAX_COUNT = 10


def sample_features_grid(
    gray: np.ndarray, 
    mask: np.ndarray, 
    grid_size: Tuple[int, int] = (10, 8), 
    max_per_cell: int = 40,
    quality_level: float = DEFAULT_QUALITY_LEVEL,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    block_size: int = DEFAULT_BLOCK_SIZE
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
    """Optical Flow処理を行うROSノード
    
    SAMマスクを受信して、オプティカルフローによる物体トラッキングを実行し、
    結果を描画して配信する。
    """
    
    def __init__(self):
        super().__init__('optflow_node')
        
        # パラメータの初期化
        self._init_parameters()
        
        # 内部状態の初期化
        self._init_state()
        
        # ROS通信の設定
        self._init_ros_communication()
        
        self.get_logger().info("Optical Flow Mask Node 起動完了")
    
    def _init_parameters(self) -> None:
        """パラメータの宣言と取得"""
        # Tracking reset parameters
        self.reset_interval = self.declare_and_get_param('reset_interval', DEFAULT_RESET_INTERVAL)
        
        # Tracking target parameters
        tracking_targets_str = self.declare_and_get_param('tracking_targets', "")
        self.tracking_targets = self._parse_tracking_targets(tracking_targets_str)
        
        # Grid sampling parameters
        self.grid_size = (
            self.declare_and_get_param('grid_size_y', DEFAULT_GRID_SIZE_Y),
            self.declare_and_get_param('grid_size_x', DEFAULT_GRID_SIZE_X)
        )
        self.max_per_cell = self.declare_and_get_param('max_per_cell', DEFAULT_MAX_PER_CELL)
        
        # Feature detection parameters
        self.quality_level = self.declare_and_get_param('quality_level', DEFAULT_QUALITY_LEVEL)
        self.min_distance = self.declare_and_get_param('min_distance', DEFAULT_MIN_DISTANCE)
        self.block_size = self.declare_and_get_param('block_size', DEFAULT_BLOCK_SIZE)
        
        # Tracking visualization parameters
        self.tracking_circle_radius = self.declare_and_get_param('tracking_circle_radius', DEFAULT_TRACKING_CIRCLE_RADIUS)
        self.tracking_circle_color = self.declare_and_get_param('tracking_circle_color', DEFAULT_TRACKING_CIRCLE_COLOR)
        
        # Optical flow parameters
        self.optical_flow_win_size = (
            self.declare_and_get_param('optical_flow_win_size_x', DEFAULT_OPTICAL_FLOW_WIN_SIZE_X),
            self.declare_and_get_param('optical_flow_win_size_y', DEFAULT_OPTICAL_FLOW_WIN_SIZE_Y)
        )
        self.optical_flow_max_level = self.declare_and_get_param('optical_flow_max_level', DEFAULT_OPTICAL_FLOW_MAX_LEVEL)
        self.optical_flow_criteria_eps = self.declare_and_get_param('optical_flow_criteria_eps', DEFAULT_OPTICAL_FLOW_CRITERIA_EPS)
        self.optical_flow_criteria_max_count = self.declare_and_get_param('optical_flow_criteria_max_count', DEFAULT_OPTICAL_FLOW_CRITERIA_MAX_COUNT)
        
        # Optical flow criteria object
        self.optical_flow_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.optical_flow_criteria_max_count,
            self.optical_flow_criteria_eps
        )
    
    def _init_state(self) -> None:
        """内部状態の初期化"""
        self.bridge = CvBridge()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts_per_label: Dict[str, np.ndarray] = {}
        self.tracking_masks_per_label: Dict[str, np.ndarray] = {}
        self.last_published_masks: Dict[str, np.ndarray] = {}  # 最後に表示したトラッキング結果を保持
        self.latest_sam_data: Optional[Dict] = None
        self.frame_count = 0
        self.sam_msg_count = 0  # SAMマスクメッセージの受信回数
    
    def _init_ros_communication(self) -> None:
        """ROS通信の設定"""
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10
        )
        self.mask_sub = self.create_subscription(
            SamMasks, '/sam_masks', self.mask_callback, 10
        )
        self.pub = self.create_publisher(ROSImage, '/image_optflow', 10)

    def declare_and_get_param(self, name: str, default_value) -> any:
        """パラメータを宣言して取得するヘルパー関数
        
        Args:
            name: パラメータ名
            default_value: デフォルト値
            
        Returns:
            パラメータの値
        """
        self.declare_parameter(name, default_value)
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

    def mask_callback(self, msg: SamMasks) -> None:
        """SAMマスクを受信して内部データを更新
        
        Args:
            msg: SamMasksメッセージ
        """
        try:
            if not msg.masks:
                self.get_logger().warn("受信したSamMasksメッセージにマスクが含まれていません")
                return
                
            self.latest_sam_data = {
                'labels': msg.labels,
                'masks': self._convert_mask_messages(msg.masks)
            }
            
            # SAMマスクメッセージの受信回数をカウント
            self.sam_msg_count += 1
            
        except Exception as e:
            self.get_logger().error(f"mask_callback エラー: {repr(e)}")
    
    def _convert_mask_messages(self, mask_msgs: List) -> List[np.ndarray]:
        """マスクメッセージをOpenCV形式に変換
        
        Args:
            mask_msgs: マスクメッセージのリスト
            
        Returns:
            OpenCV形式のマスクリスト
        """
        masks = []
        for mask_msg in mask_msgs:
            try:
                mask_cv = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
                masks.append(mask_cv)
            except Exception as e:
                self.get_logger().error(f"マスク変換エラー: {repr(e)}")
        return masks

    def image_callback(self, msg: ROSImage) -> None:
        """メイン画像処理：特徴点の初期化・追跡・マスク生成
        
        Args:
            msg: 画像メッセージ
        """
        try:
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if image_cv is None or image_cv.size == 0:
                self.get_logger().warn("受信した画像が空です")
                return
                
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

            is_reset = self._should_reset_tracking()
            if is_reset:
                self._reset_tracking_points(gray)
            else:
                self._track_points(gray)

            # トラッキング結果を表示
            if not is_reset and self.tracking_masks_per_label:
                # 通常時：現在のトラッキング結果を表示
                self._publish_tracking_result(image_cv)
                # 表示した結果を保存
                self.last_published_masks = self.tracking_masks_per_label.copy()
            elif is_reset and self.last_published_masks:
                # リセット時：前回のトラッキング結果を表示
                self._publish_last_tracking_result(image_cv)

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"image_callback エラー: {repr(e)}")

    def _should_reset_tracking(self) -> bool:
        """トラッキングの初期化が必要かを判定
        
        Returns:
            初期化が必要な場合True
        """
        return (
            self.sam_msg_count % self.reset_interval == 0 or 
            not self.prev_pts_per_label
        )

    def _reset_tracking_points(self, gray: np.ndarray) -> None:
        """特徴点を初期化
        
        Args:
            gray: グレースケール画像
        """
        if self.latest_sam_data is None:
            return
            
        self.prev_pts_per_label = {}
        self.tracking_masks_per_label = {}
        
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
                self.tracking_masks_per_label[label] = combined_mask.copy()
        
        self.prev_gray = gray
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
                # トラッキング結果をマスクに描画
                tracking_mask = self._create_tracking_mask(gray.shape, new_pts, status)
                self.tracking_masks_per_label[label] = tracking_mask
                
                # 有効な特徴点のみを保持
                valid_pts = new_pts[status == 1].reshape(-1, 1, 2)
                self.prev_pts_per_label[label] = valid_pts
        
        self.prev_gray = gray.copy()
    
    def _create_tracking_mask(
        self, 
        shape: Tuple[int, int], 
        points: np.ndarray, 
        status: np.ndarray
    ) -> np.ndarray:
        """トラッキング結果からマスクを作成
        
        Args:
            shape: マスクの形状
            points: 追跡された特徴点
            status: 各特徴点の有効性
            
        Returns:
            トラッキングマスク
        """
        tracking_mask = np.zeros(shape, dtype=np.uint8)
        
        for pt, valid in zip(points, status):
            if valid:
                x, y = pt.ravel()
                cv2.circle(
                    tracking_mask, 
                    (int(x), int(y)), 
                    self.tracking_circle_radius, 
                    self.tracking_circle_color, 
                    -1
                )
        
        return tracking_mask

    def _publish_tracking_result(self, image_cv: np.ndarray) -> None:
        """トラッキング結果を描画して配信
        
        Args:
            image_cv: 入力画像
        """
        if not self.tracking_masks_per_label:
            return
            
        detection_data = self._prepare_detection_data(image_cv.shape[:2])
        
        if detection_data['masks']:
            try:
                annotated_img = draw_image(
                    image_cv,
                    np.array(detection_data['masks']),
                    np.array(detection_data['boxes']),
                    np.array(detection_data['probs']),
                    detection_data['labels']
                )
                
                ros_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='rgb8')
                self.pub.publish(ros_msg)
                
            except Exception as e:
                self.get_logger().error(f"描画・配信エラー: {repr(e)}")
    
    def _prepare_detection_data(self, image_shape: Tuple[int, int]) -> Dict:
        """描画用のデータを準備
        
        Args:
            image_shape: 画像の形状 (height, width)
            
        Returns:
            描画用データの辞書
        """
        masks, labels, boxes, probs = [], [], [], []
        
        for label, mask in self.tracking_masks_per_label.items():
            if not np.any(mask > 0):
                continue
                
            masks.append((mask > 0).astype(np.float32))
            labels.append(f'{label}_tracking')
            boxes.append(self._calculate_bounding_box(mask, image_shape))
            probs.append(1.0)
        
        return {
            'masks': masks,
            'labels': labels,
            'boxes': boxes,
            'probs': probs
        }
    
    def _calculate_bounding_box(
        self, 
        mask: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> List[int]:
        """マスクから境界ボックスを計算
        
        Args:
            mask: マスク画像
            image_shape: 画像の形状 (height, width)
            
        Returns:
            境界ボックス [x1, y1, x2, y2]
        """
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) > 0 and len(x_indices) > 0:
            x1, y1 = x_indices.min(), y_indices.min()
            x2, y2 = x_indices.max(), y_indices.max()
            return [x1, y1, x2, y2]
        else:
            # フォールバック: 画像全体
            return [0, 0, image_shape[1], image_shape[0]]
    
    def _publish_last_tracking_result(self, image_cv: np.ndarray) -> None:
        """前回のトラッキング結果を表示（リセット時用）
        
        Args:
            image_cv: 入力画像
        """
        if not self.last_published_masks:
            return
            
        detection_data = self._prepare_detection_data_from_masks(
            self.last_published_masks, image_cv.shape[:2]
        )
        
        if detection_data['masks']:
            try:
                annotated_img = draw_image(
                    image_cv,
                    np.array(detection_data['masks']),
                    np.array(detection_data['boxes']),
                    np.array(detection_data['probs']),
                    detection_data['labels']
                )
                
                ros_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='rgb8')
                self.pub.publish(ros_msg)
                
            except Exception as e:
                self.get_logger().error(f"前回結果描画・配信エラー: {repr(e)}")
    
    def _prepare_detection_data_from_masks(
        self, 
        masks_dict: Dict[str, np.ndarray], 
        image_shape: Tuple[int, int]
    ) -> Dict:
        """マスク辞書から描画用データを準備
        
        Args:
            masks_dict: ラベルをキーとするマスク辞書
            image_shape: 画像の形状 (height, width)
            
        Returns:
            描画用データの辞書
        """
        masks, labels, boxes, probs = [], [], [], []
        
        for label, mask in masks_dict.items():
            if not np.any(mask > 0):
                continue
                
            masks.append((mask > 0).astype(np.float32))
            labels.append(f'{label}_tracking')
            boxes.append(self._calculate_bounding_box(mask, image_shape))
            probs.append(1.0)
        
        return {
            'masks': masks,
            'labels': labels,
            'boxes': boxes,
            'probs': probs
        }