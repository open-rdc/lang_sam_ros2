import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from PIL import Image
import numpy as np
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from lang_sam import LangSAM
from lang_sam.utils import draw_image


class LangSAMNode(Node):
    def __init__(self):
        super().__init__('lang_sam_node')

        # =====================================
        # パラメータの宣言と取得
        # =====================================
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'car. wheel.')

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value

        self.get_logger().info(f"使用するSAMモデル: {self.sam_model}")
        self.get_logger().info(f"使用するText Prompt: {self.text_prompt}")

        # =====================================
        # モデル、ツール、スレッド、同期
        # =====================================
        self.model = LangSAM(sam_type=self.sam_model)
        self.bridge = CvBridge()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.predicting = False
        self.lock = threading.Lock()

        # =====================================
        # ROS通信設定（サブスクライバ・パブリッシャ）
        # =====================================
        self.image_sub = self.create_subscription(
            ROSImage, '/image', self.image_callback, 10
        )
        self.mask_pub = self.create_publisher(
            ROSImage, '/image_mask', 10
        )
        self.flow_pub = self.create_publisher(
            ROSImage, '/flow_debug', 10
        )

        # =====================================
        # 前フレームの情報保持（補間用）
        # =====================================
        self.prev_image = None
        self.prev_masks = None
        self.prev_labels = None
        self.prev_boxes = None
        self.prev_scores = None
        self.reset_required = True  # 初回は必ずLangSAMを使う

        self.get_logger().info("LangSAMNode（マスク補間あり）起動完了")

    def image_callback(self, msg):
        try:
            # =====================================
            # ROS Image → OpenCV画像（RGB）
            # =====================================
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("受信画像が空です")
                return

            # 書き込み可能な形式へ変換
            cv_image = cv_image.astype(np.uint8, copy=True)

            # OpenCV (RGB) → PIL Image (RGB)
            image_pil = Image.fromarray(cv_image, mode='RGB')

        except Exception as e:
            self.get_logger().error(f"cv_bridgeまたは画像変換エラー: {e}")
            return

        # =====================================
        # LangSAM 推論を非同期で実行
        # =====================================
        with self.lock:
            if not self.predicting:
                self.predicting = True            
                self.thread_pool.submit(
                    self.run_inference,
                    image_pil,
                    cv_image.copy()
                )

        # =====================================
        # オプティカルフローによるマスクとバウンディングボックスの補間
        # =====================================
        with self.lock:
            if self.prev_image is not None and self.prev_masks is not None:
                # 前フレームと現在フレームをグレースケールに変換
                prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

                flow_vis = cv_image.copy()  # 光フロー可視化用の画像

                updated_masks = []
                updated_boxes = []

                # 各マスクと対応するバウンディングボックスに対して処理
                for i, (mask, box) in enumerate(zip(self.prev_masks, self.prev_boxes)):
                    # マスクをuint8に変換（特徴点抽出やワーピング用）
                    mask_uint8 = (mask.astype(np.uint8)) * 255

                    # マスク内の特徴点を検出
                    prev_pts = cv2.goodFeaturesToTrack(
                        prev_gray,
                        mask=mask_uint8,
                        maxCorners=200,
                        qualityLevel=0.03,
                        minDistance=5
                    )

                    # 特徴点が検出できなければマスクとボックスはそのまま
                    if prev_pts is None or len(prev_pts) < 3:
                        updated_masks.append(mask)
                        updated_boxes.append(box)
                        continue

                    prev_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=mask_uint8)
                    curr_masked = cv2.bitwise_and(curr_gray, curr_gray, mask=mask_uint8)

                    # Lucas-Kanade法で現在フレーム上の対応点を追跡
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_masked, curr_masked, prev_pts, None)

                    # 有効な追跡点のみ抽出
                    prev_valid = prev_pts[status.flatten() == 1]
                    next_valid = next_pts[status.flatten() == 1]

                    if len(prev_valid) < 3:
                        updated_masks.append(mask)
                        updated_boxes.append(box)
                        continue

                    # アフィン変換を推定（RANSACで外れ値除去）
                    M, inliers = cv2.estimateAffinePartial2D(prev_valid, next_valid, method=cv2.RANSAC, ransacReprojThreshold=3.0)

                    if M is not None:
                        # マスクをアフィン変換
                        warped_mask = cv2.warpAffine(
                            mask_uint8,
                            M,
                            (cv_image.shape[1], cv_image.shape[0]),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        warped_mask_bool = warped_mask > 0  # bool型に戻す
                        updated_masks.append(warped_mask_bool)

                        # バウンディングボックスもアフィン変換
                        x1, y1, x2, y2 = box
                        corners = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
                        transformed = cv2.transform(corners, M)
                        x1_new, y1_new = transformed[0][0]
                        x2_new, y2_new = transformed[1][0]
                        updated_boxes.append([x1_new, y1_new, x2_new, y2_new])
                    else:
                        # アフィン変換に失敗した場合はそのまま
                        updated_masks.append(mask)
                        updated_boxes.append(box)

                    # 可視化用：特徴点の動きを矢印で描画
                    for old, new, st in zip(prev_pts, next_pts, status):
                        if st:
                            x_old, y_old = old.ravel()
                            x_new, y_new = new.ravel()
                            cv2.arrowedLine(
                                flow_vis,
                                (int(x_old), int(y_old)),
                                (int(x_new), int(y_new)),
                                color=(0, 255, 0),
                                thickness=1,
                                tipLength=0.3
                            )

                # flow_vis パブリッシュ
                try:
                    flow_msg = self.bridge.cv2_to_imgmsg(flow_vis, encoding='rgb8')
                    self.flow_pub.publish(flow_msg)
                except Exception as e:
                    self.get_logger().error(f"フロー描画画像のパブリッシュ失敗: {e}")

                # 更新されたマスクで描画
                masks_np = np.array(updated_masks)
                if len(updated_boxes) == 0:
                    updated_boxes = np.zeros((0, 4), dtype=np.float32)  # safety fallback

                annotated_image = draw_image(
                    image_rgb=cv_image,
                    masks=masks_np,
                    xyxy=np.array(updated_boxes),
                    probs=self.prev_scores,
                    labels=self.prev_labels
                )
                # 更新マスクを保存して次に使う
                self.prev_masks = masks_np
                self.prev_boxes = np.array(updated_boxes)
            else:
                annotated_image = cv_image.copy()

        # =====================================
        # 処理済み画像のパブリッシュ
        # =====================================
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            self.mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"マスク画像のパブリッシュ失敗: {e}")

    def run_inference(self, image_pil, image_np):
        try:
            start = time.time()
            results = self.model.predict([image_pil], [self.text_prompt])
            elapsed = time.time() - start
            # self.get_logger().info(f"LangSAM 推論完了: {elapsed:.2f}秒")

            masks = np.array(results[0]['masks'])
            boxes = np.array(results[0]['boxes'])
            scores = np.array(results[0]['scores'])
            labels = results[0]['labels']

            with self.lock:
                self.prev_image = image_np.copy()

                # ❗LangSAMの結果を使うのは初回かリセット時のみ
                if self.reset_required:
                    self.prev_masks = masks.copy()
                    self.prev_boxes = boxes.copy()
                    self.prev_scores = scores.copy()
                    self.prev_labels = labels.copy()
                    self.reset_required = False

        except Exception as e:
            self.get_logger().error(f"LangSAM推論エラー: {e}")
        finally:
            self.predicting = False


def main(args=None):
    rclpy.init(args=args)
    node = LangSAMNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
