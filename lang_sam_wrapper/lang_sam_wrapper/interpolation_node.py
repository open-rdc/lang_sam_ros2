import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from lang_sam_msgs.msg import SamMasks
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from lang_sam.utils import draw_image
import threading


class InterpolationNode(Node):
    """
    interpolationノード
    
    optflow_nodeから出力されるトラッキングデータ(/sam_masks)から
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
        self.declare_parameter('input_topic', '/sam_masks')
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
        # SAMマスクデータのサブスクライバー（独立したコールバックグループ）
        self.sam_masks_sub = self.create_subscription(
            SamMasks,
            self.input_topic,
            self.sam_masks_callback,
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
        self.latest_sam_masks = None
        
        # トラッキング補間用の状態
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
    
    def sam_masks_callback(self, msg: SamMasks):
        """
        SAMマスクデータのコールバック - トラッキング点のみ更新
        
        Args:
            msg: SAMマスクメッセージ
        """
        with self.processing_lock:
            self.latest_sam_masks = msg
            self.get_logger().debug(f"Received SAM masks: {len(msg.labels)} labels")
            # トラッキング点のみ更新（画像配信はimage_callbackで行う）
            self._update_tracking_points_from_sam_masks(msg)
    
    def _publish_image_with_tracking(self):
        """
        元画像にトラッキング点を重ねて配信（画像ベース）
        """
        if self.latest_image is None:
            return
        
        try:
            # 画像をCV2形式に変換
            image_cv = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='rgb8')
            
            # 現在のトラッキング点があれば描画
            if self.last_valid_tracking_points:
                annotated_image = self._draw_simple_tracking_points(image_cv, self.last_valid_tracking_points)
            else:
                # トラッキング点がない場合は元画像をそのまま
                annotated_image = image_cv.copy()
            
            # ROSメッセージとして配信
            ros_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'camera_frame'
            self.interpolated_pub.publish(ros_msg)
            
            # デバッグ情報（30フレームごとに出力）
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 1
            
            if self.frame_count % 30 == 0:
                point_count = sum(len(points) for points in self.last_valid_tracking_points.values())
                self.get_logger().info(
                    f"Published image with {point_count} tracking points to {self.output_topic}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error in _publish_image_with_tracking: {repr(e)}")
    
    def _update_tracking_points_from_sam_masks(self, sam_masks_msg: SamMasks):
        """
        SAMマスクからトラッキング点を高速で更新（軽量版）
        
        Args:
            sam_masks_msg: SAMマスクメッセージ
        """
        try:
            new_tracking_points = {}
            
            for i, label in enumerate(sam_masks_msg.labels):
                # "_tracking"で終わるラベルのみ処理
                if label.endswith('_tracking') and i < len(sam_masks_msg.masks):
                    mask_msg = sam_masks_msg.masks[i]
                    mask_cv = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
                    
                    # 高速な特徴点抽出（マスクから直接座標を取得）
                    points = self._fast_extract_points_from_mask(mask_cv)
                    if len(points) > 0:
                        clean_label = label.replace('_tracking', '')
                        new_tracking_points[clean_label] = points
            
            # 新しいデータがあれば更新
            if new_tracking_points:
                self.last_valid_tracking_points = new_tracking_points
                self.get_logger().debug(f"Updated tracking points: {len(new_tracking_points)} labels")
            
        except Exception as e:
            self.get_logger().error(f"Error in _update_tracking_points_from_sam_masks: {repr(e)}")
    
    def _fast_extract_points_from_mask(self, mask: np.ndarray):
        """
        マスクから特徴点を高速抽出
        
        Args:
            mask: バイナリマスク
            
        Returns:
            np.ndarray: 特徴点の配列
        """
        # 白い部分の座標を直接取得（最も高速）
        y_coords, x_coords = np.where(mask > 128)
        
        if len(x_coords) == 0:
            return np.array([])
        
        # サンプリングして点数を制限（パフォーマンス向上）
        if len(x_coords) > 50:  # 最大50点に制限
            indices = np.random.choice(len(x_coords), 50, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
        
        # [x, y]の形式で返す
        points = np.column_stack((x_coords, y_coords))
        return points
    
    
    def _draw_simple_tracking_points(self, image: np.ndarray, tracking_points_dict):
        """
        画像にトラッキング点を描画（シンプル版）
        
        Args:
            image: 入力画像
            tracking_points_dict: {label: points} の辞書
            
        Returns:
            np.ndarray: 描画済み画像
        """
        annotated_image = image.copy()
        
        # 色の定義（異なるラベルごとに異なる色）
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue (BGR format)
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, (label, points) in enumerate(tracking_points_dict.items()):
            color = colors[i % len(colors)]
            
            # 各点を円で描画
            for point in points:
                cv2.circle(
                    annotated_image, 
                    tuple(point.astype(int)), 
                    self.tracking_circle_radius, 
                    color, 
                    -1
                )
            
            # ラベルを描画（最初の点の近くに）
            if len(points) > 0:
                label_pos = (int(points[0][0]) + 10, int(points[0][1]) - 10)
                cv2.putText(
                    annotated_image,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        return annotated_image


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