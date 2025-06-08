import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np


def sample_features_grid(gray, mask, grid_size=(10, 8), max_per_cell=40):
    # -------------
    # グリッド単位で特徴点を抽出する関数
    # -------------
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
                for pt in p:
                    pt[0][0] += x0
                    pt[0][1] += y0
                all_pts.extend(p)

    return np.array(all_pts) if all_pts else None


class OptFlowMaskNode(Node):
    def __init__(self):
        super().__init__('optflow_mask_node')

        # -------------
        # パラメータ宣言・取得
        # -------------
        self.declare_parameter('reset_interval', 60)
        self.declare_parameter('grid_size_x', 10)
        self.declare_parameter('grid_size_y', 8)
        self.declare_parameter('max_per_cell', 20)

        self.reset_interval = self.get_parameter('reset_interval').value
        self.grid_size = (
            self.get_parameter('grid_size_y').value,
            self.get_parameter('grid_size_x').value
        )
        self.max_per_cell = self.get_parameter('max_per_cell').value

        # -------------
        # 初期化処理
        # -------------
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = None
        self.mask = None
        self.frame_count = 0

        # -------------
        # ROS 2 通信設定（購読・配信）
        # -------------
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10)
        self.mask_sub = self.create_subscription(
            ROSImage, '/image_mask', self.mask_callback, 10)
        self.pub = self.create_publisher(ROSImage, '/image_opt_mask', 10)

        self.latest_mask_cv = None
        self.get_logger().info("Optical Flow Mask Node 起動完了")

    def mask_callback(self, msg):
        # -------------
        # マスク画像を受信して保存
        # -------------
        try:
            mask_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            self.latest_mask_cv = (gray_mask > 10).astype(np.uint8) * 255
            self.get_logger().info("マスク画像を受信・更新しました")
        except Exception as e:
            self.get_logger().error(f"mask_callback エラー: {e}")

    def image_callback(self, msg):
        # -------------
        # メイン画像処理：特徴点の更新・追跡・マスク生成
        # -------------
        try:
            image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

            if self.frame_count % self.reset_interval == 0 or self.prev_pts is None:
                if self.latest_mask_cv is not None:
                    p0 = sample_features_grid(
                        gray, self.latest_mask_cv, self.grid_size, self.max_per_cell
                    )
                    if p0 is not None:
                        self.get_logger().info(f"特徴点数: {len(p0)}")
                    else:
                        self.get_logger().warn("特徴点が検出されませんでした")
                    self.prev_pts = p0
                    self.prev_gray = gray
                    self.mask = self.latest_mask_cv.copy()

            elif self.prev_pts is not None and self.prev_gray is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prevImg=self.prev_gray,
                    nextImg=gray,
                    prevPts=self.prev_pts,
                    nextPts=None,
                    winSize=(15, 15),
                    maxLevel=2,
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

            if self.mask is not None:
                out_img = image_cv.copy()
                out_img[self.mask > 0] = [255, 0, 0]
                ros_img = self.bridge.cv2_to_imgmsg(out_img, encoding='rgb8')
                self.pub.publish(ros_img)
                self.get_logger().info("/image_opt_mask にマスク付き画像を送信しました")

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"image_callback エラー: {e}")
