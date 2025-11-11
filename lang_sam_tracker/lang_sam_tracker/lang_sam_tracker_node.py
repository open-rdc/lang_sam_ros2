#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image as ROSImage

from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
import numpy as np
import cv2

from lang_sam import LangSAM
from lang_sam.utils import draw_image
from lang_sam.models.utils import DEVICE

class LangSamTrackerNode(Node):
    def __init__(self):
        super().__init__('lang_sam_tracker')
        self.logger = self.get_logger()
        self.logger.info('Initializing LangSAM Tracker Node...')

        # パラメータの読み込み
        self._setup_parameters()

        # deviceの設定
        self.device = DEVICE

        # LangSAMモデルの初期化
        self.model = LangSAM()

        self.bridge = CvBridge()

        # トラッキング用状態
        self.tracks = []          # [{id, label, score, points (Nx1x2), box [x1,y1,x2,y2]}]
        self.prev_gray = None
        self.next_track_id = 0

        # 検出の時間管理（秒ベース）
        self.last_detection_time: Time | None = None

        # ROS 2のサブスクライバとパブリッシャの設定
        self.image_sub = self.create_subscription(ROSImage, '/camera/image_raw', self.image_callback, 1)
        # 2系統の出力
        self.image_detection_pub = self.create_publisher(ROSImage, '/image/lang_sam/detection', 1)
        self.image_tracking_pub = self.create_publisher(ROSImage, '/image/lang_sam/tracking', 1)

        # ログの設定
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Using SAM model: {self.sam_model}')
        self.get_logger().info(f'Using text prompt: {self.text_prompt}')
        self.get_logger().info(f'Detection interval (sec): {self.detection_interval_sec}')
        self.get_logger().info('LangSAM model initialized.')

    # パラメータ取得用のヘルパーメソッド
    def _setup_parameters(self):
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'wheel. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('detection_interval_sec', 2.0)

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.detection_interval_sec = self.get_parameter('detection_interval_sec').get_parameter_value().double_value

    def _init_tracks_from_detections(self, cv_image, boxes, labels, scores, masks_bool):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.tracks = []
        h, w = gray.shape
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            # LangSAMのマスクを使用（bool -> uint8）
            if i < masks_bool.shape[0]:
                mask_uint8 = (masks_bool[i].astype(np.uint8)) * 255
            else:
                mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(mask_uint8, (x1, y1), (x2, y2), 255, -1)

            # goodFeaturesToTrackでマスク内の特徴点抽出
            pts = cv2.goodFeaturesToTrack(
                image=gray,
                maxCorners=120,
                qualityLevel=0.01,
                minDistance=3,
                mask=mask_uint8
            )
            if pts is None or pts.shape[0] < 5:
                continue

            track = {
                'id': self.next_track_id,
                'label': labels[i] if i < len(labels) else 'obj',
                'score': float(scores[i]) if i < len(scores) else 1.0,
                'points': pts,              # (N,1,2) float32
                'box': [x1, y1, x2, y2],
                'mask': masks_bool[i] if i < masks_bool.shape[0] else (mask_uint8.astype(bool))
            }
            self.tracks.append(track)
            self.next_track_id += 1

    def _mask_from_points(self, points, shape):
        # points: (M,1,2)
        if points is None or points.shape[0] < 3:
            return np.zeros(shape, dtype=bool)
        pts_2d = points.reshape(-1, 2)
        hull = cv2.convexHull(pts_2d.astype(np.float32))
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(int), 255)
        return mask.astype(bool)

    def _update_tracks_with_klt(self, cv_image):
        if self.prev_gray is None or len(self.tracks) == 0:
            return
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        new_tracks = []
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        )
        h, w = gray.shape
        for track in self.tracks:
            pts = track['points'].astype(np.float32)
            new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **lk_params)
            if new_pts is None or st is None:
                continue

            # 有効点抽出
            st_flat = st.flatten().astype(bool)
            good = new_pts[st_flat]

            # 形状を(N,2)に正規化
            if good.ndim == 3 and good.shape[1] == 1 and good.shape[2] == 2:
                good_xy = good[:, 0, :]              # (N,2)
            elif good.ndim == 2 and good.shape[1] == 2:
                good_xy = good                        # (N,2)
            else:
                try:
                    good_xy = good.reshape(-1, 2)
                except Exception:
                    continue

            if good_xy.shape[0] < 5:
                continue

            # bbox更新
            x_min = int(np.clip(np.min(good_xy[:, 0]), 0, w - 1))
            y_min = int(np.clip(np.min(good_xy[:, 1]), 0, h - 1))
            x_max = int(np.clip(np.max(good_xy[:, 0]), 0, w - 1))
            y_max = int(np.clip(np.max(good_xy[:, 1]), 0, h - 1))

            # マスク再構成（凸包）
            good_pts = good_xy.reshape(-1, 1, 2).astype(np.float32)
            mask_bool = self._mask_from_points(good_pts, (h, w))

            track['points'] = good_pts
            track['box'] = [x_min, y_min, x_max, y_max]
            track['mask'] = mask_bool
            new_tracks.append(track)

        self.tracks = new_tracks
        self.prev_gray = gray

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        now = self.get_clock().now()
        need_detection = False
        if self.last_detection_time is None:
            need_detection = True
        else:
            if (now - self.last_detection_time) >= Duration(seconds=float(self.detection_interval_sec)):
                need_detection = True

        if need_detection:
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                results = self.model.predict([pil_image], [self.text_prompt])
            det = results[0]

            boxes = det.get('boxes', None)
            if boxes is None:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
            else:
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu().numpy()
                boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

            h, w, _ = cv_image.shape
            masks = det.get('masks', None)
            if masks is None:
                masks_np = np.zeros((boxes_np.shape[0], h, w), dtype=bool)
            else:
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()
                masks_np = np.asarray(masks)
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0]
                if masks_np.ndim == 3:
                    pass
                elif masks_np.ndim == 2:
                    masks_np = masks_np[None, ...]
                else:
                    masks_np = np.zeros((boxes_np.shape[0], h, w), dtype=bool)
                masks_np = masks_np.astype(bool)

            labels = det.get('labels', [])
            labels_det = [str(l) for l in labels] if len(labels) > 0 else []
            scores = det.get('scores', None)
            if scores is None:
                scores_np = np.zeros((boxes_np.shape[0],), dtype=np.float32)
            else:
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)

            # 検出イメージを/detection
            det_image_pil = draw_image(
                image_rgb=pil_image,
                masks=masks_np,
                xyxy=boxes_np,
                probs=scores_np,
                labels=labels_det,
            )
            det_image_cv = cv2.cvtColor(np.array(det_image_pil), cv2.COLOR_RGB2BGR)
            det_msg = self.bridge.cv2_to_imgmsg(det_image_cv, encoding='bgr8')
            self.image_detection_pub.publish(det_msg)

            # トラック初期化 (マスクを渡す)
            self._init_tracks_from_detections(
                cv_image,
                boxes_np.tolist(),
                labels_det,
                scores_np.tolist(),
                masks_np
            )
            self.last_detection_time = now
        else:
            self._update_tracks_with_klt(cv_image)

        # トラッキング結果を/tracking
        if self.tracks:
            boxes_for_draw = np.asarray([t['box'] for t in self.tracks], dtype=np.int32)
            labels_for_draw = [t['label'] for t in self.tracks]
            scores_for_draw = np.asarray([t['score'] for t in self.tracks], dtype=np.float32)
            masks_for_draw = np.asarray([t['mask'] for t in self.tracks], dtype=bool)
        else:
            h, w, _ = cv_image.shape
            boxes_for_draw = np.zeros((0, 4), dtype=np.int32)
            labels_for_draw = []
            scores_for_draw = np.zeros((0,), dtype=np.float32)
            masks_for_draw = np.zeros((0, h, w), dtype=bool)

        pil_base = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        try:
            track_image_pil = draw_image(
                image_rgb=pil_base,
                masks=masks_for_draw,
                xyxy=boxes_for_draw,
                probs=scores_for_draw,
                labels=labels_for_draw,
            )
        except Exception as e:
            self.get_logger().warn(f'draw_image失敗(tracking): {e}')
            return

        track_image_cv = cv2.cvtColor(np.array(track_image_pil), cv2.COLOR_RGB2BGR)
        track_msg = self.bridge.cv2_to_imgmsg(track_image_cv, encoding='bgr8')
        self.image_tracking_pub.publish(track_msg)


def main():
    rclpy.init()
    node = LangSamTrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
