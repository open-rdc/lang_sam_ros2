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
from lang_sam.utils import draw_image  # 可視化ユーティリティ
from lang_sam.models.utils import DEVICE  # 推論デバイス（cuda/cpu取得）
from lang_sam_msgs.msg import TrackArray, Track # カスタムメッセージ

class LangSamTrackerNode(Node):
    def __init__(self):
        super().__init__('lang_sam_tracker')
        self.logger = self.get_logger()
        self.logger.info('Initializing LangSAM Tracker Node...')

        # パラメータ宣言・取得
        self._setup_parameters()

        # 使用デバイス（外部ユーティリティから自動選択）
        self.device = DEVICE

        # LangSAMモデルのロード（GroundingDINO+SAMの複合推論）
        self.model = LangSAM()

        # ROS <-> OpenCV画像変換
        self.bridge = CvBridge()

        # KLTトラッキング状態
        # tracks: 各トラックの状態を辞書で保持
        #  - points: KLT特徴点 (N,1,2) float32
        #  - box:    推定bbox [x1,y1,x2,y2]（KLT点のmin/maxから更新）
        #  - mask:   特徴点の凸包から再構成したboolマスク（可視化用）
        #  - label/score: 可視化の補助情報
        self.tracks = []
        self.prev_gray = None     # 直前のグレースケール（KLT入力）
        self.next_track_id = 0

        # 最終検出時刻（検出間隔secを満たしたらLangSAM再実行）
        self.last_detection_time: Time | None = None

        # I/O: 入力画像サブスク / 出力画像パブリッシュ
        self.image_sub = self.create_subscription(ROSImage, '/camera/image_raw', self.image_callback, 1)
        self.image_detection_pub = self.create_publisher(ROSImage, '/image/lang_sam/detection', 1)
        self.image_tracking_pub = self.create_publisher(ROSImage, '/image/lang_sam/tracking', 1)
        self.tracks_pub = self.create_publisher(TrackArray, '/lang_sam/tracks', 10)

        # ログ
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Using SAM model: {self.sam_model}')
        self.get_logger().info(f'Using text prompt: {self.text_prompt}')
        self.get_logger().info(f'Detection interval (sec): {self.detection_interval_sec}')
        self.get_logger().info('LangSAM model initialized.')

    # パラメータ取得用の関数
    def _setup_parameters(self):
        # 注意: launch/config側で上書き可能
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'wheel. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('detection_interval_sec', 2.0)

        # KLT(LK光学フロー)のROSパラメータ
        # - 窓サイズ、ピラミッド段数、収束条件、最低存続点数
        self.declare_parameter('klt_win_size', [15, 15])      # integer_array [w, h]
        self.declare_parameter('klt_max_level', 3)            # integer
        self.declare_parameter('klt_criteria_count', 30)      # integer
        self.declare_parameter('klt_criteria_eps', 0.03)      # double
        self.declare_parameter('klt_min_points', 5)           # integer: 維持すべき最小追跡点数

        # GFTT(Shi-Tomasi)のROSパラメータ
        self.declare_parameter('gftt_max_corners', 120)       # integer
        self.declare_parameter('gftt_quality_level', 0.01)    # double
        self.declare_parameter('gftt_min_distance', 3.0)      # double(画素)

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.detection_interval_sec = self.get_parameter('detection_interval_sec').get_parameter_value().double_value

        # KLTパラメータの取得と整形
        ws = self.get_parameter('klt_win_size').get_parameter_value().integer_array_value
        self.klt_win_size = (int(ws[0]), int(ws[1])) if len(ws) >= 2 else (15, 15)
        self.klt_max_level = int(self.get_parameter('klt_max_level').get_parameter_value().integer_value)
        self.klt_criteria_count = int(self.get_parameter('klt_criteria_count').get_parameter_value().integer_value)
        self.klt_criteria_eps = float(self.get_parameter('klt_criteria_eps').get_parameter_value().double_value)
        self.klt_min_points = int(self.get_parameter('klt_min_points').get_parameter_value().integer_value)

        # GFTTパラメータの取得
        self.gftt_max_corners = int(self.get_parameter('gftt_max_corners').get_parameter_value().integer_value)
        self.gftt_quality_level = float(self.get_parameter('gftt_quality_level').get_parameter_value().double_value)
        self.gftt_min_distance = float(self.get_parameter('gftt_min_distance').get_parameter_value().double_value)

    def _init_tracks_from_detections(self, cv_image, boxes, labels, scores, masks_bool):
        # 検出結果からトラック群を初期化
        # - マスク領域からgoodFeaturesToTrackでKLTの初期点をサンプリング
        # - bbox/label/score/maskをtrack辞書に格納
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.tracks = []
        h, w = gray.shape
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]

            # マスクが無い検出はトラックを生成しない（疑似マスクは作らない設計）
            if i >= masks_bool.shape[0] or masks_bool[i].dtype != bool or not masks_bool[i].any():
                continue
            mask_uint8 = (masks_bool[i].astype(np.uint8)) * 255  # goodFeaturesToTrackがuint8マスクを要求

            # GFTTパラメータをROSから取得した値で適用
            pts = cv2.goodFeaturesToTrack(
                image=gray,
                maxCorners=self.gftt_max_corners,
                qualityLevel=self.gftt_quality_level,
                minDistance=self.gftt_min_distance,
                mask=mask_uint8
            )
            # 最低点数をROSパラメータで判定
            if pts is None or pts.shape[0] < self.klt_min_points:
                continue

            track = {
                'id': self.next_track_id,
                'label': labels[i] if i < len(labels) else 'obj',
                'score': float(scores[i]) if i < len(scores) else 1.0,
                'points': pts,              # (N,1,2) float32
                'box': [x1, y1, x2, y2],
                'mask': masks_bool[i]       # 検出時点のマスク（bool, HxW）
            }
            self.tracks.append(track)
            self.next_track_id += 1

    def _mask_from_points(self, points, shape):
        # KLT更新後の特徴点群から凸包を生成し、それを塗りつぶしてマスクを再構成
        # points: (M,1,2)、shape: (H,W)
        if points is None or points.shape[0] < 3:
            # 凸包が作れない（点が少ない）場合は空マスク
            return np.zeros(shape, dtype=bool)
        pts_2d = points.reshape(-1, 2)
        hull = cv2.convexHull(pts_2d.astype(np.float32))  # 凸包頂点
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(int), 255)   # 凸包内部を塗りつぶし
        return mask.astype(bool)

    def _update_tracks_with_klt(self, cv_image):
        # 直前(prev_gray)と現在フレームの間でピラミッドLKを計算し、各トラックの特徴点/マスク/bboxを更新
        if self.prev_gray is None or len(self.tracks) == 0:
            return
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        new_tracks = []
        # KLTパラメータをROSから取得した値で適用
        lk_params = dict(
            winSize=tuple(map(int, self.klt_win_size)),
            maxLevel=int(self.klt_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      int(self.klt_criteria_count),
                      float(self.klt_criteria_eps))
        )
        h, w = gray.shape
        for track in self.tracks:
            pts = track['points'].astype(np.float32)  # (N,1,2)
            new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **lk_params)
            if new_pts is None or st is None:
                continue

            # 追跡成功点のみ抽出
            st_flat = st.flatten().astype(bool)
            good = new_pts[st_flat]

            # 形状を(N,2)に正規化
            if good.ndim == 3 and good.shape[1] == 1 and good.shape[2] == 2:
                good_xy = good[:, 0, :]
            elif good.ndim == 2 and good.shape[1] == 2:
                good_xy = good
            else:
                try:
                    good_xy = good.reshape(-1, 2)
                except Exception:
                    continue

            # 最低点数をROSパラメータで判定
            if good_xy.shape[0] < self.klt_min_points:
                continue

            # bboxは特徴点のmin/maxから更新（画像境界でクリップ）
            x_min = int(np.clip(np.min(good_xy[:, 0]), 0, w - 1))
            y_min = int(np.clip(np.min(good_xy[:, 1]), 0, h - 1))
            x_max = int(np.clip(np.max(good_xy[:, 0]), 0, w - 1))
            y_max = int(np.clip(np.max(good_xy[:, 1]), 0, h - 1))

            # 特徴点の凸包からマスクを再構成（可視化で利用）
            good_pts = good_xy.reshape(-1, 1, 2).astype(np.float32)
            mask_bool = self._mask_from_points(good_pts, (h, w))

            # トラック更新
            track['points'] = good_pts
            track['box'] = [x_min, y_min, x_max, y_max]
            track['mask'] = mask_bool
            new_tracks.append(track)

        # 次フレームのKLTに備えてprev_gray更新
        self.tracks = new_tracks
        self.prev_gray = gray

    def image_callback(self, msg):
        # 入力: ROS Image -> OpenCV(BGR)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 検出再実行の判定（nodeのClock基準、秒ベース）
        now = self.get_clock().now()
        need_detection = False
        if self.last_detection_time is None:
            need_detection = True
        else:
            if (now - self.last_detection_time) >= Duration(seconds=float(self.detection_interval_sec)):
                need_detection = True

        if need_detection:
            # 検出フレーム: LangSAM推論（PIL RGBに変換して入力）
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                results = self.model.predict([pil_image], [self.text_prompt])
            det = results[0]

            # boxes/masks/scores/labelsをnumpyへ正規化（空でも次工程のshapeが破綻しないように）
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
                # (N,1,H,W)->(N,H,W) などに整形し、boolへキャスト
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0]
                if masks_np.ndim == 3:
                    pass
                elif masks_np.ndim == 2:
                    masks_np = masks_np[0:1, ...]
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

            # 検出結果を可視化して/detectionにパブリッシュ
            det_image_pil = draw_image(
                image_rgb=pil_image,      # PIL RGB
                masks=masks_np,           # (N,H,W) bool
                xyxy=boxes_np,            # (N,4) float32
                probs=scores_np,          # (N,) float32
                labels=labels_det,        # list[str]
            )
            det_image_cv = cv2.cvtColor(np.array(det_image_pil), cv2.COLOR_RGB2BGR)
            det_msg = self.bridge.cv2_to_imgmsg(det_image_cv, encoding='bgr8')
            self.image_detection_pub.publish(det_msg)

            # 検出マスクを用いてKLT初期点を生成し、トラックを初期化
            self._init_tracks_from_detections(
                cv_image,
                boxes_np.tolist(),
                labels_det,
                scores_np.tolist(),
                masks_np
            )
            self.last_detection_time = now
        else:
            # トラッキングフレーム: KLTで特徴点を更新し、凸包マスクを再構成
            self._update_tracks_with_klt(cv_image)

        # 毎フレーム、トラッキング結果を/trackingに可視化・配信
        if self.tracks:
            # numpy配列に正規化
            boxes_for_draw = np.asarray([t['box'] for t in self.tracks], dtype=np.int32)
            labels_for_draw = [t['label'] for t in self.tracks]                                 # list[str]
            scores_for_draw = np.asarray([t['score'] for t in self.tracks], dtype=np.float32)   # (N,)
            masks_for_draw = np.asarray([t['mask'] for t in self.tracks], dtype=bool)           # (N,H,W)
        else:
            # 空でもshapeを満たすダミー配列を渡す（utils側のshape検証回避）
            h, w, _ = cv_image.shape
            boxes_for_draw = np.zeros((0, 4), dtype=np.int32)
            labels_for_draw = []
            scores_for_draw = np.zeros((0,), dtype=np.float32)
            masks_for_draw = np.zeros((0, h, w), dtype=bool)

        # 背景は現フレーム（RGB）
        pil_base = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        try:
            track_image_pil = draw_image(
                image_rgb=pil_base,
                masks=masks_for_draw,   # KLT再構成マスク（bool, (N,H,W)）
                xyxy=boxes_for_draw,    # (N,4)
                probs=scores_for_draw,  # (N,)
                labels=labels_for_draw, # list[str]
            )
        except Exception as e:
            # 型・shape不一致等の可視化例外をログ化（処理はスキップ）
            self.get_logger().warn(f'draw_image失敗(tracking): {e}')
            return

        # PIL(RGB) -> OpenCV(BGR) -> ROS Image
        track_image_cv = cv2.cvtColor(np.array(track_image_pil), cv2.COLOR_RGB2BGR)
        track_msg = self.bridge.cv2_to_imgmsg(track_image_cv, encoding='bgr8')
        self.image_tracking_pub.publish(track_msg)

        # トラック情報を/custom_msgs/TrackArrayで配信
        msg_tracks = TrackArray()
        msg_tracks.header.stamp = self.get_clock().now().to_msg()
        msg_tracks.header.frame_id = 'camera'
        for t in self.tracks:
            tr = Track()
            tr.id = int(t['id'])
            tr.label = str(t['label'])
            tr.score = float(t['score'])
            x1, y1, x2, y2 = t['box']
            tr.x_min = int(x1); tr.y_min = int(y1)
            tr.x_max = int(x2); tr.y_max = int(y2)
            msg_tracks.tracks.append(tr)
        self.tracks_pub.publish(msg_tracks)


def main(args=None):
    rclpy.init(args=args)
    node = LangSamTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
