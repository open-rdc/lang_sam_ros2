#!/usr/bin/env python3
"""
LangSAM Tracker Node with Native C++ CSRT Implementation (Synchronous Version)
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

from lang_sam.lang_sam_tracker import LangSAMTracker
from lang_sam.models.utils import DEVICE
from lang_sam.tracker_utils.csrt_client import CSRTClient
from lang_sam.tracker_utils.config_manager import ConfigManager
from lang_sam.tracker_utils.logging_manager import setup_logging_for_ros_node
from lang_sam.utils import draw_image


class Detection:
    """検出結果クラス"""
    def __init__(self, box, label, score=1.0):
        self.x = int(box[0])
        self.y = int(box[1])
        self.width = int(box[2] - box[0])
        self.height = int(box[3] - box[1])
        self.label = label
        self.score = score


class LangSAMTrackerNode(Node):
    """Main LangSAM node with C++ CSRT tracker integration (Synchronous Version)"""
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        # 統一ロギングシステム初期化
        self.logger, self.perf_logger, self.error_ctx = setup_logging_for_ros_node(self, debug=False)
        self.logger.info("LangSAMトラッカーノード（C++ CSRT版）同期処理で初期化開始")
        
        # 統一設定管理システム初期化
        try:
            config_manager = ConfigManager.get_instance()
            config_path = "/home/ryusei/formula_ws/src/lang_sam_executor/config/config.yaml"
            self.system_config = config_manager.load_from_yaml(config_path, "lang_sam_tracker_node")
            self.logger.info("統一設定管理システム初期化完了")
        except Exception as e:
            self.logger.error(f"設定読み込み失敗、デフォルト設定を使用: {e}")
            self._declare_parameters()
            self._load_parameters()
            self.system_config = None
        
        # Initialize CV Bridge
        self.cv_bridge = CvBridge()
        
        # Initialize LangSAM tracker
        if self.system_config:
            sam_model = self.system_config.model.sam_model
            text_prompt = self.system_config.model.text_prompt
            box_threshold = self.system_config.model.box_threshold
            text_threshold = self.system_config.model.text_threshold
            gdino_interval_seconds = self.system_config.execution.gdino_interval_seconds
            input_topic = self.system_config.ros.input_topic
            gdino_topic = self.system_config.ros.gdino_topic
            csrt_output_topic = self.system_config.ros.csrt_output_topic
            sam_topic = self.system_config.ros.sam_topic
            tracking_targets = self.system_config.model.tracking_targets
            bbox_margin = self.system_config.tracking.bbox_margin
            bbox_min_size = self.system_config.tracking.bbox_min_size
        else:
            sam_model = self.sam_model
            text_prompt = self.text_prompt
            box_threshold = self.box_threshold
            text_threshold = self.text_threshold
            gdino_interval_seconds = self.gdino_interval_seconds
            input_topic = self.input_topic
            gdino_topic = self.gdino_topic
            csrt_output_topic = self.csrt_output_topic
            sam_topic = self.sam_topic
            tracking_targets = self.tracking_targets
            bbox_margin = self.bbox_margin
            bbox_min_size = self.bbox_min_size
            
        # Initialize native C++ CSRT client
        self.csrt_client = CSRTClient(self)
        
        if not self.csrt_client.is_available():
            self.logger.error("ネイティブC++ CSRT拡張が利用できません、終了します")
            return
            
        self.logger.info("ネイティブC++ CSRTクライアント初期化完了")
        
        # Initialize LangSAM tracker
        self.lang_sam_tracker = LangSAMTracker(
            sam_type=sam_model,
            device=str(DEVICE)
        )
        
        # Store configuration
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.gdino_interval_seconds = gdino_interval_seconds
        self.tracking_targets = tracking_targets
        self.bbox_margin = bbox_margin
        self.bbox_min_size = bbox_min_size
        
        # Timing control
        self.last_gdino_time = 0.0
        self.frame_count = 0
        
        # Publishers
        self.gdino_pub = self.create_publisher(Image, gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, csrt_output_topic, 10)
        self.sam_pub = self.create_publisher(Image, sam_topic, 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        # Current tracking state
        self.current_detection_labels = []
        
        self.logger.info("LangSAMトラッカーノード初期化完了（同期処理版）")
    
    def _declare_parameters(self):
        """Declare parameters (fallback)"""
        self.declare_parameter('sam_model', 'sam2.1_hiera_tiny')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.25)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('gdino_interval_seconds', 1.0)
        
        # Topic parameters
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_output_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # Tracking parameters
        self.declare_parameter('bbox_margin', 5)
        self.declare_parameter('bbox_min_size', 3)
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
    
    def _load_parameters(self):
        """Load parameters from ROS parameter server"""
        self.sam_model = self.get_parameter('sam_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        self.gdino_interval_seconds = self.get_parameter('gdino_interval_seconds').value
        
        self.input_topic = self.get_parameter('input_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.csrt_output_topic = self.get_parameter('csrt_output_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
        
        self.tracking_targets = self.get_parameter('tracking_targets').value
        self.bbox_margin = self.get_parameter('bbox_margin').value
        self.bbox_min_size = self.get_parameter('bbox_min_size').value
    
    def image_callback(self, msg: Image):
        """同期画像処理コールバック"""
        with self.error_ctx.handle_errors("image_processing", "ros_callback", reraise=False):
            # Convert ROS image to OpenCV
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            self.frame_count += 1
            
            # GroundingDINO検出（同期処理）
            should_run_gdino = (current_time - self.last_gdino_time) >= self.gdino_interval_seconds
            
            if should_run_gdino:
                self.logger.info(f"フレーム{self.frame_count}: GroundingDINO同期検出開始")
                detections = self._run_grounding_dino_detection(image)
                
                if detections:
                    self.logger.info(f"フレーム{self.frame_count}: 検出結果適用 {len(detections)}オブジェクト")
                    self._apply_detection_results(detections, image, msg.header)
                
                # GroundingDINO可視化配信
                self._publish_gdino_visualization(detections, image, msg.header)
                
                self.last_gdino_time = current_time
            
            # CSRTトラッキング処理（常に実行）
            self._process_csrt_tracking(image, msg.header)
    
    def _run_grounding_dino_detection(self, image: np.ndarray) -> list:
        """GroundingDINO同期検出実行"""
        try:
            from PIL import Image as PILImage
            
            # 画像変換
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            start_time = time.time()
            
            # GroundingDINO推論実行
            results = self.lang_sam_tracker.predict_with_tracking(
                images_pil=[pil_image],
                texts_prompt=[self.text_prompt],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                update_trackers=False,
                run_sam=True
            )
            
            processing_time = time.time() - start_time
            
            # 検出結果解析
            detections = self._parse_detection_results(results)
            
            self.logger.info(f"GroundingDINO同期処理完了: {len(detections)}オブジェクト ({processing_time:.3f}秒)")
            return detections
            
        except Exception as e:
            self.logger.error(f"GroundingDINO検出エラー: {e}")
            return []
    
    def _parse_detection_results(self, results) -> list:
        """検出結果解析"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            
            if len(boxes) > 0 and len(labels) > 0:
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    score = scores[i] if i < len(scores) else 1.0
                    detection = Detection(box, label, score)
                    detections.append(detection)
        
        return detections
    
    def _apply_detection_results(self, detections: list, current_image: np.ndarray, header):
        """検出結果の適用処理"""
        # C++ CSRTトラッカー初期化
        detection_boxes = []
        detection_labels = []
        
        for det in detections:
            if hasattr(det, 'label') and det.label in self.tracking_targets:
                bbox = (det.x, det.y, det.width, det.height)
                detection_boxes.append(bbox)
                detection_labels.append(det.label)
        
        if detection_boxes:
            # C++ CSRT初期化
            csrt_results = self.csrt_client.process_detections(
                current_image, detection_boxes, detection_labels
            )
            self.logger.info(f"C++ CSRT初期化完了: {len(csrt_results)}トラッカー")
            
            # 検出結果ラベル保存
            self.current_detection_labels = detection_labels.copy()
    
    def _process_csrt_tracking(self, image: np.ndarray, header):
        """CSRT継続トラッキング処理"""
        # CSRT更新
        csrt_results = self.csrt_client.update_trackers(image)
        
        if csrt_results:
            # CSRTラベル取得
            tracker_labels = self.csrt_client.get_cached_labels()
            
            # CSRT可視化生成・配信
            self._publish_csrt_visualization(csrt_results, tracker_labels, image, header)
            
            # SAM2セグメンテーション
            self._process_sam_segmentation(csrt_results, tracker_labels, image, header)
        else:
            # トラッカーなし、空画像配信
            self._publish_empty_images(image, header)
    
    def _publish_gdino_visualization(self, detections: list, image: np.ndarray, header):
        """GroundingDINO可視化配信"""
        try:
            if not detections:
                # 検出なしの場合は元画像をそのまま配信
                gdino_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                gdino_msg.header = header
                self.gdino_pub.publish(gdino_msg)
                return
            
            # draw_image用データ準備
            xyxy = []
            labels = []
            probs = []
            
            for detection in detections:
                x1, y1 = detection.x, detection.y
                x2, y2 = x1 + detection.width, y1 + detection.height
                xyxy.append([x1, y1, x2, y2])
                labels.append(detection.label)
                probs.append(detection.score)
            
            # draw_imageで可視化
            gdino_img = draw_image(
                image_rgb=image,
                masks=None,
                xyxy=np.array(xyxy),
                probs=probs,
                labels=labels
            )
            
            # ROS配信
            gdino_msg = self.cv_bridge.cv2_to_imgmsg(gdino_img, 'bgr8')
            gdino_msg.header = header
            self.gdino_pub.publish(gdino_msg)
            
        except Exception as e:
            self.logger.error(f"GroundingDINO可視化エラー: {e}")
    
    def _publish_csrt_visualization(self, csrt_results: list, labels: list, 
                                  image: np.ndarray, header):
        """CSRT可視化配信"""
        try:
            if not csrt_results:
                csrt_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                csrt_msg.header = header
                self.csrt_pub.publish(csrt_msg)
                return
            
            # draw_image用データ準備
            xyxy = []
            for bbox in csrt_results:
                x, y, w, h = bbox
                xyxy.append([x, y, x + w, y + h])
            
            # draw_imageで可視化
            csrt_img = draw_image(
                image_rgb=image,
                masks=None,
                xyxy=np.array(xyxy),
                probs=[1.0] * len(xyxy),
                labels=labels[:len(xyxy)]
            )
            
            # ROS配信
            csrt_msg = self.cv_bridge.cv2_to_imgmsg(csrt_img, 'bgr8')
            csrt_msg.header = header
            self.csrt_pub.publish(csrt_msg)
            
        except Exception as e:
            self.logger.error(f"CSRT可視化エラー: {e}")
    
    def _process_sam_segmentation(self, csrt_results: list, labels: list, 
                                image: np.ndarray, header):
        """SAM2セグメンテーション処理"""
        try:
            if not csrt_results:
                return
            
            # SAM2推論実行
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # CSRTバウンディングボックスをSAM2に渡す
            boxes = []
            for bbox in csrt_results:
                x, y, w, h = bbox
                boxes.append([x, y, x + w, y + h])
            
            if boxes:
                # SAM2セグメンテーション実行
                sam_img = self._run_sam2_segmentation(image, pil_image, boxes, labels)
                
                # ROS配信
                sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_img, 'bgr8')
                sam_msg.header = header
                self.sam_pub.publish(sam_msg)
            else:
                # ボックスなし、元画像配信
                sam_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                sam_msg.header = header
                self.sam_pub.publish(sam_msg)
            
        except Exception as e:
            self.logger.warning(f"SAM2推論エラー: {e}")
            # エラー時は元画像を配信
            sam_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            sam_msg.header = header
            self.sam_pub.publish(sam_msg)
    
    def _run_sam2_segmentation(self, image: np.ndarray, pil_image, boxes: list, labels: list):
        """SAM2セグメンテーション実行 (draw_image使用)"""
        try:
            # ModelCoordinatorのSAMにアクセス
            sam_predictor = self.lang_sam_tracker.coordinator.sam
            
            if sam_predictor is None:
                self.logger.warning("SAM2 predictor not available")
                return self._create_bbox_visualization(image, boxes, labels)
            
            # RGB画像に変換（SAM2はRGB入力を期待）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # SAM2推論実行（各ボックスに対してセグメンテーション）
            xyxy = []
            masks = []
            valid_labels = []
            
            for i, box in enumerate(boxes):
                try:
                    # xyxy形式に変換
                    x1, y1, x2, y2 = box
                    xyxy_box = np.array([x1, y1, x2, y2])
                    
                    # SAM2推論
                    sam_masks, scores, logits = sam_predictor.predict(image_rgb, xyxy_box)
                    
                    if sam_masks is not None and len(sam_masks) > 0:
                        # 最高スコアのマスクを使用
                        best_mask = sam_masks[0]  # multimask_output=Falseなので1つだけ
                        
                        # マスクを2D配列に変換
                        if len(best_mask.shape) == 3:
                            best_mask = best_mask[0]  # 次元を削減
                        
                        # draw_image用にデータを準備
                        xyxy.append([x1, y1, x2, y2])
                        masks.append(best_mask)
                        valid_labels.append(labels[i] if i < len(labels) else "unknown")
                        
                except Exception as e:
                    self.logger.debug(f"SAM2推論エラー (box {i}): {e}")
                    continue
            
            if xyxy and masks:
                # draw_imageでセグメンテーション結果を可視化
                sam_img = draw_image(
                    image_rgb=image,
                    masks=masks,
                    xyxy=np.array(xyxy),
                    probs=[1.0] * len(xyxy),
                    labels=valid_labels
                )
                return sam_img
            else:
                # SAM2結果なし、バウンディングボックス表示にフォールバック
                return self._create_bbox_visualization(image, boxes, labels)
            
        except Exception as e:
            self.logger.warning(f"SAM2セグメンテーションエラー: {e}")
            return self._create_bbox_visualization(image, boxes, labels)
    
    def _get_label_color(self, label: str):
        """ラベル別の色を取得"""
        colors = {
            'white line': [0, 255, 0],      # 緑
            'red pylon': [0, 0, 255],       # 赤
            'car': [255, 0, 0],             # 青
            'human': [255, 255, 0],         # シアン
        }
        return colors.get(label, [255, 255, 255])  # デフォルトは白
    
    def _create_bbox_visualization(self, image: np.ndarray, boxes: list, labels: list):
        """バウンディングボックス可視化（SAM2の代替表示）"""
        try:
            # draw_image用データ準備
            xyxy = []
            for box in boxes:
                if len(box) == 4:
                    # 既にxyxy形式の場合
                    xyxy.append(box)
                else:
                    # xywh形式の場合
                    x, y, w, h = box
                    xyxy.append([x, y, x + w, y + h])
            
            if xyxy:
                # draw_imageで可視化
                sam_img = draw_image(
                    image_rgb=image,
                    masks=None,
                    xyxy=np.array(xyxy),
                    probs=[1.0] * len(xyxy),
                    labels=labels[:len(xyxy)]
                )
                return sam_img
            else:
                return image
            
        except Exception as e:
            self.logger.warning(f"ボックス可視化エラー: {e}")
            return image
    
    def _publish_empty_images(self, image: np.ndarray, header):
        """トラッカーなし時の空画像配信"""
        # 元画像をそのまま全トピックに配信
        image_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
        image_msg.header = header
        
        self.csrt_pub.publish(image_msg)
        self.sam_pub.publish(image_msg)
    
    def __del__(self):
        """デストラクタ"""
        self.logger.info("LangSAMトラッカーノード終了処理開始")
        
        if hasattr(self, 'csrt_client'):
            self.csrt_client.clear_trackers()
        
        self.logger.info("LangSAMトラッカーノード終了処理完了")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LangSAMTrackerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Node error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()