import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from lang_sam_msgs.msg import FeaturePoints
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
from lang_sam.utils import draw_image
import threading


class InterpolationNode(Node):
    """
    interpolationノード
    
    optflow_nodeから出力されるトラッキングデータ(/image_optflow_features)から
    トラッキング点のみを抽出して画像に描画するノード
    """
    
    def __init__(self):
        super().__init__('interpolation_node')
        
        # コールバックグループの設定（非同期処理のため別々のグループ）
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.sam_callback_group = MutuallyExclusiveCallbackGroup()
        self.publisher_callback_group = ReentrantCallbackGroup()
        
        # スレッドセーフティ用ロック
        self.processing_lock = threading.Lock()
        
        # CV Bridge初期化
        self.bridge = CvBridge()
        
        # パラメータ設定
        self._init_parameters()
        
        # ROS通信の初期化
        self._init_ros_communication()
        
        self.get_logger().info("InterpolationNode initialized")
    
    def _init_parameters(self):
        """パラメータの初期化"""
        # 入力トピック名
        self.declare_parameter('input_topic', '/image_optflow_features')
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        
        # 出力トピック名
        self.declare_parameter('output_topic', '/sam_masks_interpolated')
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # 元画像トピック名
        self.declare_parameter('image_topic', '/image')
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        
        # トラッキング点の描画パラメータ
        self.declare_parameter('tracking_circle_radius', 3)
        self.tracking_circle_radius = self.get_parameter('tracking_circle_radius').value
        
        self.declare_parameter('tracking_circle_color', [0, 255, 0])  # Green
        self.tracking_circle_color = self.get_parameter('tracking_circle_color').value
        
        # QoSサイズ
        self.declare_parameter('qos_size', 10)
        self.qos_size = self.get_parameter('qos_size').value
        
        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Output topic: {self.output_topic}")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"QoS size: {self.qos_size}")
    
    def _init_ros_communication(self):
        """ROS通信の初期化"""
        # FeaturePointsデータのサブスクライバー（独立したコールバックグループ）
        self.feature_points_sub = self.create_subscription(
            FeaturePoints,
            self.input_topic,
            self.feature_points_callback,
            self.qos_size,
            callback_group=self.sam_callback_group
        )
        
        # 元画像のサブスクライバー（独立したコールバックグループ）
        self.image_sub = self.create_subscription(
            ROSImage,
            self.image_topic,
            self.image_callback,
            self.qos_size,
            callback_group=self.image_callback_group
        )
        
        # 補間された画像のパブリッシャー
        self.interpolated_pub = self.create_publisher(
            ROSImage,
            self.output_topic,
            self.qos_size,
            callback_group=self.publisher_callback_group
        )
        
        # 内部状態の初期化
        self.latest_image = None
        self.last_valid_tracking_points = {}  # 最後の有効なトラッキング点を保持
        
        self.get_logger().info("ROS communication initialized")
    
    def image_callback(self, msg: ROSImage):
        """
        画像データのコールバック - 画像が来るたびに即座に配信
        
        Args:
            msg: 画像メッセージ
        """
        with self.processing_lock:
            self.latest_image = msg
            self.get_logger().debug(f"Received image: {msg.width}x{msg.height}")
            # 画像ベースで即座に配信（SAMマスクを待たない）
            self._publish_image_with_tracking()
    
    def feature_points_callback(self, msg: FeaturePoints):
        """
        FeaturePointsデータのコールバック - トラッキング点のみ更新
        
        Args:
            msg: FeaturePointsメッセージ
        """
        with self.processing_lock:
            self._update_tracking_points_from_feature_points(msg)
    
    def _publish_image_with_tracking(self):
        """
        元画像にトラッキング点を重ねて配信（draw_imageベース）
        """
        if self.latest_image is None:
            return
        
        try:
            # 画像をCV2形式に変換
            image_cv = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='rgb8')
            
            # 現在のトラッキング点があれば描画
            if self.last_valid_tracking_points:
                annotated_image = self._publish_image_with_draw_image(image_cv)
            else:
                annotated_image = image_cv.copy()
            
            # ROSメッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.interpolated_pub.publish(ros_msg)
            
                
        except Exception as e:
            self.get_logger().error(f"Error in _publish_image_with_tracking: {repr(e)}")
    
    def _update_tracking_points_from_feature_points(self, feature_points_msg: FeaturePoints):
        """
        FeaturePointsメッセージからトラッキング点を高速で更新
        
        Args:
            feature_points_msg: FeaturePointsメッセージ
        """
        try:
            new_tracking_points = {}
            
            # 各ラベルごとに特徴点を取得
            point_index = 0
            for i, label in enumerate(feature_points_msg.labels):
                if i < len(feature_points_msg.point_counts):
                    point_count = feature_points_msg.point_counts[i]
                    
                    # 該当する特徴点を取得
                    if point_index + point_count <= len(feature_points_msg.points):
                        label_points = feature_points_msg.points[point_index:point_index + point_count]
                        
                        # Point32からnumpy配列に変換
                        points = np.array([[p.x, p.y] for p in label_points])
                        
                        if len(points) > 0:
                            clean_label = label.replace('_tracking', '')
                            new_tracking_points[clean_label] = points
                        
                        point_index += point_count
            
            # 新しいデータがあれば更新
            if new_tracking_points:
                self.last_valid_tracking_points = new_tracking_points
            
        except Exception as e:
            self.get_logger().error(f"Error in _update_tracking_points_from_feature_points: {repr(e)}")
    
    
    
    def _publish_image_with_draw_image(self, image_cv: np.ndarray) -> np.ndarray:
        """
        トラッキング結果をdraw_imageを使って描画（OptFlowNodeと同じスタイル）
        
        Args:
            image_cv: 入力画像
            
        Returns:
            np.ndarray: 描画済み画像
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
                
                return annotated_img
            else:
                return image_cv.copy()
                
        except Exception as e:
            self.get_logger().error(f"draw_image描画エラー: {repr(e)}")
            return image_cv.copy()
    
    def _prepare_tracking_detection_data(self, image_shape):
        """
        トラッキング点から描画用データを準備（OptFlowNodeと同じロジック）
        
        Args:
            image_shape: 画像の形状 (height, width)
            
        Returns:
            描画用データの辞書
        """
        masks = []
        labels = []
        boxes = []
        probs = []
        
        for label, pts in self.last_valid_tracking_points.items():
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
    
    def _create_tracking_point_mask(self, pts: np.ndarray, image_shape) -> np.ndarray:
        """
        トラッキング点から小さなマスクを作成
        
        Args:
            pts: トラッキング点
            image_shape: 画像の形状
            
        Returns:
            マスク
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for pt in pts:
            x, y = pt.ravel() if len(pt.shape) > 1 else pt
            cv2.circle(mask, (int(x), int(y)), self.tracking_circle_radius, 255, -1)
        
        return mask
    
    def _calculate_bounding_box_from_points(self, pts: np.ndarray, image_shape):
        """
        トラッキング点からバウンディングボックスを計算
        
        Args:
            pts: トラッキング点
            image_shape: 画像の形状
            
        Returns:
            バウンディングボックス [x1, y1, x2, y2]
        """
        if len(pts) == 0:
            return [0, 0, 0, 0]
        
        # 点の座標を取得（2次元配列形式に対応）
        if len(pts.shape) > 1 and pts.shape[1] == 2:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
        else:
            x_coords = [pt[0] for pt in pts]
            y_coords = [pt[1] for pt in pts]
        
        # 余裕を持たせてバウンディングボックスを計算
        margin = self.tracking_circle_radius * 2
        x1 = max(0, min(x_coords) - margin)
        y1 = max(0, min(y_coords) - margin)
        x2 = min(image_shape[1], max(x_coords) + margin)
        y2 = min(image_shape[0], max(y_coords) + margin)
        
        return [float(x1), float(y1), float(x2), float(y2)]


def main(args=None):
    """メイン関数"""
    rclpy.init(args=args)
    
    try:
        node = InterpolationNode()
        
        # マルチスレッド実行器を使用
        executor = MultiThreadedExecutor(num_threads=2)
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