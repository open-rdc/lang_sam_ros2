import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np


# -------------
# グリッドごとに特徴点を抽出する関数
# -------------
def sample_features_grid(gray, mask, grid_size=(10, 8), max_per_cell=40):
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

            p = cv2.goodFeaturesToTrack(
                roi_gray,
                mask=roi_mask,
                maxCorners=max_per_cell,
                qualityLevel=0.001,
                minDistance=1,
                blockSize=3
            )

            if p is not None:
                p[:, 0, 0] += x0
                p[:, 0, 1] += y0
                all_pts.extend(p)

    return np.array(all_pts) if all_pts else None


# -------------
# Optical Flow 処理ノードクラス
# -------------
class OptFlowNode(Node):
    def __init__(self):
        super().__init__('optflow_node')

        # -------------
        # パラメータ宣言・取得
        # -------------
        self.reset_interval = self.declare_and_get_param('reset_interval', 60)
        self.grid_size = (
            self.declare_and_get_param('grid_size_y', 8),
            self.declare_and_get_param('grid_size_x', 10)
        )
        self.max_per_cell = self.declare_and_get_param('max_per_cell', 20)

        # -------------
        # 内部状態の初期化
        # -------------
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = None
        self.mask = None
        self.latest_mask_cv = None
        self.frame_count = 0

        # -------------
        # ROS 2 通信設定
        # -------------
        self.image_sub = self.create_subscription(ROSImage, '/image', self.image_callback, 10)
        self.mask_sub = self.create_subscription(ROSImage, '/image_sam', self.mask_callback, 10)
        self.pub = self.create_publisher(ROSImage, '/image_optflow', 10)

        self.get_logger().info("Optical Flow Mask Node 起動完了")

    # -------------
    # パラメータ取得ヘルパー関数
    # -------------
    def declare_and_get_param(self, name, default_value):
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).value

    # -------------
    # マスク画像を受信して保存
    # -------------
    def mask_callback(self, msg):
        try:
            mask_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            self.latest_mask_cv = (gray_mask > 10).astype(np.uint8) * 255
        except Exception as e:
            self.get_logger().error(f"mask_callback エラー: {repr(e)}")

    # -------------
    # メイン画像処理：特徴点の初期化・追跡・マスク生成
    # -------------
    def image_callback(self, msg):
        try:
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

            if self.need_reset():
                self.reset_tracking_points(gray)
            else:
                self.track_points(gray)

            if self.mask is not None:
                self.publish_mask_overlay(image_cv)

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"image_callback エラー: {repr(e)}")

    # -------------
    # 初期化の必要条件判定
    # -------------
    def need_reset(self):
        return self.frame_count % self.reset_interval == 0 or self.prev_pts is None

    # -------------
    # 特徴点を初期化
    # -------------
    def reset_tracking_points(self, gray):
        if self.latest_mask_cv is not None:
            p0 = sample_features_grid(
                gray, self.latest_mask_cv, self.grid_size, self.max_per_cell
            )
            self.prev_pts = p0
            self.prev_gray = gray
            self.mask = self.latest_mask_cv.copy()

    # -------------
    # Optical Flow による特徴点の追跡
    # -------------
    def track_points(self, gray):
        if self.prev_pts is not None and self.prev_gray is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            if p1 is not None and st is not None:
                self.mask = np.zeros_like(gray)
                for pt, valid in zip(p1, st):
                    if valid:
                        x, y = pt.ravel()
                        cv2.circle(self.mask, (int(x), int(y)), 4, 255, -1)
                self.prev_pts = p1[st == 1].reshape(-1, 1, 2)
                self.prev_gray = gray.copy()

    # -------------
    # マスクを画像に重ねて配信
    # -------------
    def publish_mask_overlay(self, image_cv):
        out_img = image_cv.copy()
        out_img[self.mask > 0] = [255, 0, 0]
        ros_img = self.bridge.cv2_to_imgmsg(out_img, encoding='rgb8')
        self.pub.publish(ros_img)
