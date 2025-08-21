#!/usr/bin/env python3
"""
LangSAM Tracker Node with Native C++ CSRT Implementation
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from lang_sam.lang_sam_tracker import LangSAMTracker
from lang_sam.models.utils import DEVICE
from lang_sam.tracker_utils.csrt_client import CSRTClient
from lang_sam.tracker_utils.config_manager import ConfigManager
from lang_sam.tracker_utils.logging_manager import setup_logging_for_ros_node
from lang_sam.tracker_utils.frame_buffer import CSRTFrameManager
from lang_sam.utils import draw_image


@dataclass
class DetectionRequest:
    """GroundingDINO検出リクエスト"""
    image: np.ndarray
    pil_image: 'Image.Image'
    frame_id: int
    timestamp: float


@dataclass
class DetectionResult:
    """GroundingDINO検出結果"""
    frame_id: int
    detections: list
    timestamp: float
    processing_time: float


class AsyncDetectionManager:
    """非同期検出処理マネージャー"""
    
    def __init__(self, tracker: LangSAMTracker, text_prompt: str, 
                 box_threshold: float, text_threshold: float, logger):
        self.tracker = tracker
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.logger = logger
        
        # 非同期処理用キューとスレッド
        self.request_queue = queue.Queue(maxsize=2)  # 最大2フレーム分をキュー
        self.result_queue = queue.Queue(maxsize=5)   # 結果保持
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gdino")
        self.is_processing = threading.Event()
        self.shutdown_event = threading.Event()
        
        # バックグラウンド処理スレッド開始
        self.processing_thread = threading.Thread(target=self._background_processing, daemon=True)
        self.processing_thread.start()
    
    def submit_detection_request(self, image: np.ndarray, pil_image: 'Image.Image', 
                               frame_id: int) -> bool:
        """非同期検出リクエスト送信"""
        if self.request_queue.full():
            # キューが満杯の場合、古いリクエストを破棄
            try:
                old_request = self.request_queue.get_nowait()
                self.logger.warning(f"検出リクエストキュー満杯、フレーム{old_request.frame_id}を破棄")
            except queue.Empty:
                pass
        
        try:
            request = DetectionRequest(
                image=image.copy(),
                pil_image=pil_image,
                frame_id=frame_id,
                timestamp=time.time()
            )
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            self.logger.warning(f"検出リクエスト送信失敗: フレーム{frame_id}")
            return False
    
    def get_latest_result(self) -> Optional[DetectionResult]:
        """最新の検出結果取得"""
        latest_result = None
        try:
            # キューから利用可能な最新結果を取得
            while True:
                result = self.result_queue.get_nowait()
                if latest_result is None or result.frame_id > latest_result.frame_id:
                    latest_result = result
        except queue.Empty:
            pass
        
        return latest_result
    
    def _background_processing(self):
        """バックグラウンド検出処理ループ"""
        self.logger.info("非同期検出処理スレッド開始")
        
        while not self.shutdown_event.is_set():
            try:
                # リクエスト待機（タイムアウト付き）
                request = self.request_queue.get(timeout=1.0)
                
                self.is_processing.set()
                start_time = time.time()
                
                self.logger.debug(f"フレーム{request.frame_id}: GroundingDINO非同期処理開始")
                
                # GroundingDINO推論実行
                results = self.tracker.predict_with_tracking(
                    images_pil=[request.pil_image],
                    texts_prompt=[self.text_prompt],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    update_trackers=False,
                    run_sam=True
                )
                
                processing_time = time.time() - start_time
                
                # 検出結果解析
                detections = self._parse_detection_results(results)
                
                # 結果をキューに格納
                result = DetectionResult(
                    frame_id=request.frame_id,
                    detections=detections,
                    timestamp=request.timestamp,
                    processing_time=processing_time
                )
                
                try:
                    self.result_queue.put_nowait(result)
                    self.logger.debug(f"フレーム{request.frame_id}: 検出完了 {len(detections)}オブジェクト ({processing_time:.3f}秒)")
                except queue.Full:
                    # 結果キューが満杯の場合、古い結果を破棄
                    try:
                        old_result = self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                        self.logger.warning(f"結果キュー満杯、フレーム{old_result.frame_id}結果を破棄")
                    except queue.Empty:
                        pass
                
                self.is_processing.clear()
                
            except queue.Empty:
                # タイムアウト、継続
                continue
            except Exception as e:
                self.logger.error(f"非同期検出処理エラー: {e}")
                self.is_processing.clear()
    
    def _parse_detection_results(self, results) -> list:
        """検出結果解析（既存ロジックを移植）"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            
            if len(boxes) > 0 and len(labels) > 0:
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    class Detection:
                        def __init__(self, box, label, score=1.0):
                            self.x = int(box[0])
                            self.y = int(box[1])
                            self.width = int(box[2] - box[0])
                            self.height = int(box[3] - box[1])
                            self.label = label
                            self.score = score
                    
                    score = scores[i] if i < len(scores) else 1.0
                    detection = Detection(box, label, score)
                    detections.append(detection)
        
        return detections
    
    def is_busy(self) -> bool:
        """処理中判定"""
        return self.is_processing.is_set()
    
    def shutdown(self):
        """非同期処理終了"""
        self.logger.info("非同期検出処理終了開始")
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        self.logger.info("非同期検出処理終了完了")


class LangSAMTrackerNode(Node):
    """Main LangSAM node with C++ CSRT tracker integration"""
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        # 統一ロギングシステム初期化
        self.logger, self.perf_logger, self.error_ctx = setup_logging_for_ros_node(self, debug=False)
        self.logger.info("LangSAMトラッカーノード（C++ CSRT版）初期化開始")
        
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
        
        # 設定値の取得（統一設定管理またはレガシー）
        if self.system_config:
            self.sam_model = self.system_config.model.sam_model
            self.text_prompt = self.system_config.model.text_prompt
            self.box_threshold = self.system_config.model.box_threshold
            self.text_threshold = self.system_config.model.text_threshold
            self.gdino_interval_seconds = self.system_config.execution.gdino_interval_seconds
            self.input_topic = self.system_config.ros.input_topic
            self.gdino_topic = self.system_config.ros.gdino_topic
            self.csrt_output_topic = self.system_config.ros.csrt_output_topic
            self.sam_topic = self.system_config.ros.sam_topic
            self.tracking_targets = self.system_config.model.tracking_targets
            self.bbox_margin = self.system_config.tracking.bbox_margin
            self.bbox_min_size = self.system_config.tracking.bbox_min_size
            
            # CSRT復旧機能設定
            self.enable_csrt_recovery = getattr(self.system_config.tracking, 'enable_csrt_recovery', True)
            self.frame_buffer_duration = getattr(self.system_config.tracking, 'frame_buffer_duration', 5.0)
            self.time_travel_seconds = getattr(self.system_config.tracking, 'time_travel_seconds', 1.0)
            self.fast_forward_frames = getattr(self.system_config.tracking, 'fast_forward_frames', 10)
        else:
            # レガシー設定からCSRT復旧機能パラメータ取得
            self.declare_parameter('enable_csrt_recovery', True)
            self.declare_parameter('frame_buffer_duration', 5.0)
            self.declare_parameter('time_travel_seconds', 1.0)
            self.declare_parameter('fast_forward_frames', 10)
            
            self.enable_csrt_recovery = self.get_parameter('enable_csrt_recovery').value
            self.frame_buffer_duration = self.get_parameter('frame_buffer_duration').value
            self.time_travel_seconds = self.get_parameter('time_travel_seconds').value
            self.fast_forward_frames = self.get_parameter('fast_forward_frames').value
        
        # Initialize native C++ CSRT client
        self.csrt_client = CSRTClient(self)
        
        # CSRT復旧機能統合初期化（バッファ・時間遡行・早送り）
        if self.enable_csrt_recovery:
            self.csrt_frame_manager = CSRTFrameManager(
                buffer_duration=self.frame_buffer_duration,
                time_travel_seconds=self.time_travel_seconds,
                fast_forward_frames=self.fast_forward_frames
            )
            self.logger.info(f"CSRT復旧機能有効化: バッファ{self.frame_buffer_duration}秒, 時間遡行{self.time_travel_seconds}秒, 早送り{self.fast_forward_frames}フレーム")
        else:
            self.csrt_frame_manager = None
            self.logger.info("CSRT復旧機能無効")
        
        if not self.csrt_client.is_available():
            self.logger.error("ネイティブC++ CSRT拡張が利用できません、終了します")
            return
        
        # Initialize LangSAM tracker (detection and segmentation only)
        with self.perf_logger.measure_time("tracker_initialization"):
            self.tracker = LangSAMTracker(
                sam_type=self.sam_model,
                device=DEVICE
            )
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscriber
        queue_size = self.system_config.ros.queue_size if self.system_config else 10
        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            queue_size
        )
        
        # Timing variables
        self.last_gdino_time = 0.0
        self.frame_count = 0
        
        # Store detection labels for CSRT/SAM visualization
        self.current_detection_labels = []
        
        # 非同期検出処理マネージャー初期化
        self.async_detection_manager = AsyncDetectionManager(
            tracker=self.tracker,
            text_prompt=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            logger=self.logger
        )
        
        # Thread safety（軽量化：CSRTとSAM用のみ）
        self.processing_lock = threading.Lock()
        
        self.logger.info("LangSAMトラッカーノード（非同期並列処理版）初期化完了")
        self._log_configuration()
    
    def _declare_parameters(self):
        """Declare all ROS parameters"""
        # Core parameters
        self.declare_parameter('sam_model', 'sam2.1_hiera_tiny')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.3)
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
        
        # CSRT復旧機能パラメータ
        self.declare_parameter('enable_csrt_recovery', True)
        self.declare_parameter('frame_buffer_duration', 5.0)
        self.declare_parameter('time_travel_seconds', 3.0)
        self.declare_parameter('fast_forward_frames', 10)
        
        # All CSRT parameters are already declared by CSRTClient
    
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
        
        self.bbox_margin = self.get_parameter('bbox_margin').value
        self.bbox_min_size = self.get_parameter('bbox_min_size').value
        self.tracking_targets = self.get_parameter('tracking_targets').value
    
    def _setup_publishers(self):
        """Setup ROS publishers"""
        self.gdino_pub = self.create_publisher(Image, self.gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, self.csrt_output_topic, 10)
        self.sam_pub = self.create_publisher(Image, self.sam_topic, 10)
    
    def _log_configuration(self):
        """設定概要ログ（統一ロギング使用）"""
        self.logger.info("=== LangSAM設定概要 ===")
        self.logger.info(f"SAMモデル: {self.sam_model}")
        self.logger.info(f"テキストプロンプト: {self.text_prompt}")
        self.logger.info(f"ボックス閾値: {self.box_threshold}")
        self.logger.info(f"テキスト閾値: {self.text_threshold}")
        self.logger.info(f"GDINO実行間隔: {self.gdino_interval_seconds}秒")
        self.logger.info(f"入力トピック: {self.input_topic}")
        self.logger.info(f"追跡対象: {self.tracking_targets}")
        self.logger.info(f"C++ CSRT利用可能: {self.csrt_client.is_available()}")
        self.logger.info(f"アクティブトラッカー数: {self.csrt_client.get_tracker_count()}")
        self.logger.info(f"CSRT復旧機能: {'有効' if self.enable_csrt_recovery else '無効'}")
        
        # 復旧統計初期表示
        if self.enable_csrt_recovery:
            recovery_stats = self.csrt_client.get_recovery_stats()
            self.logger.info(f"CSRT復旧統計: 復旧成功{recovery_stats['recovered']}, 失敗{recovery_stats['failed']}, 有効{recovery_stats['recovery_enabled']}")
        
        if self.system_config:
            # パフォーマンス設定の推奨表示
            config_manager = ConfigManager.get_instance()
            recommendations = config_manager.get_performance_recommendations()
            if recommendations.get("suggestions"):
                self.logger.info("パフォーマンス推奨設定あり")
        
        self.logger.info("=== 設定概要終了 ===")
    
    def image_callback(self, msg: Image):
        """画像処理コールバック（非同期並列処理版）"""
        with self.error_ctx.handle_errors("image_processing", "ros_callback", reraise=False):
            # Convert ROS image to OpenCV（常に実行）
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            self.frame_count += 1
            
            # 非同期GroundingDINO検出の管理
            should_run_gdino = (current_time - self.last_gdino_time) >= self.gdino_interval_seconds
            
            if should_run_gdino and not self.async_detection_manager.is_busy():
                # 非同期検出リクエスト送信
                from PIL import Image as PILImage
                try:
                    pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    if self.async_detection_manager.submit_detection_request(image, pil_image, self.frame_count):
                        self.logger.debug(f"フレーム{self.frame_count}: 非同期GroundingDINO検出リクエスト送信")
                        self.last_gdino_time = current_time
                    else:
                        self.logger.warning(f"フレーム{self.frame_count}: 検出リクエスト送信失敗")
                        
                except Exception as e:
                    self.logger.error(f"画像変換失敗: {e}")
            
            # 非同期検出結果の確認と適用
            detection_result = self.async_detection_manager.get_latest_result()
            if detection_result and detection_result.detections:
                self._apply_detection_results(detection_result, image, msg.header)
            
            # CSRTトラッキング処理（常に実行、軽量）
            self._process_csrt_tracking(image, msg.header)
    
    def _apply_detection_results(self, detection_result: DetectionResult, 
                               current_image: np.ndarray, header):
        """非同期検出結果の適用処理"""
        with self.processing_lock:
            self.logger.info(f"フレーム{detection_result.frame_id}: 検出結果適用 {len(detection_result.detections)}オブジェクト")
            
            # C++ CSRTトラッカー初期化
            detection_boxes = []
            detection_labels = []
            
            for det in detection_result.detections:
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
            
            # GroundingDINO可視化画像生成・配信
            self._publish_gdino_visualization(detection_result.detections, current_image, header)
    
    def _process_csrt_tracking(self, image: np.ndarray, header):
        """CSRT継続トラッキング処理（復旧機能統合・軽量・常時実行）"""
        # フレームバッファに追加（復旧機能有効時）
        if self.csrt_frame_manager:
            frame_id = self.csrt_frame_manager.add_frame(image)
            
        # CSRT更新（復旧機能統合・GPU処理なし、軽量）
        if self.enable_csrt_recovery:
            csrt_results = self.csrt_client.update_trackers_with_recovery(image)
        else:
            csrt_results = self.csrt_client.update_trackers(image)
        
        if csrt_results:
            self.logger.debug(f"フレーム{self.frame_count}: CSRTトラッキング更新 {len(csrt_results)}オブジェクト")
            
            # CSRTラベル取得
            tracker_labels = self.csrt_client.get_cached_labels()
            
            # CSRT可視化生成・配信
            self._publish_csrt_visualization(csrt_results, tracker_labels, image, header)
            
            # SAM2セグメンテーション（GPU処理だが独立実行）
            self._process_sam_segmentation(csrt_results, tracker_labels, image, header)
        else:
            # トラッカーなし、空画像配信
            self._publish_empty_images(image, header)
    
    def _publish_gdino_visualization(self, detections: list, image: np.ndarray, header):
        """GroundingDINO可視化配信（draw_image使用）"""
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
                # xyxy形式に変換
                x1, y1 = detection.x, detection.y
                x2, y2 = x1 + detection.width, y1 + detection.height
                xyxy.append([x1, y1, x2, y2])
                labels.append(detection.label)
                probs.append(detection.score)
            
            # RGB変換してdraw_image実行
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            empty_masks = np.zeros((len(xyxy), image.shape[0], image.shape[1]), dtype=bool)
            
            gdino_image_rgb = draw_image(image_rgb, empty_masks, np.array(xyxy), np.array(probs), labels)
            
            # BGR変換してROS配信
            gdino_image_bgr = cv2.cvtColor(gdino_image_rgb, cv2.COLOR_RGB2BGR)
            gdino_msg = self.cv_bridge.cv2_to_imgmsg(gdino_image_bgr, 'bgr8')
            gdino_msg.header = header
            self.gdino_pub.publish(gdino_msg)
            
        except Exception as e:
            self.logger.error(f"GroundingDINO可視化エラー: {e}")
    
    def _publish_csrt_visualization(self, csrt_results: list, tracker_labels: list, 
                                  image: np.ndarray, header):
        """CSRT可視化配信（draw_image使用）"""
        try:
            if not csrt_results or not tracker_labels:
                # トラッキング結果なしの場合は元画像をそのまま配信
                csrt_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                csrt_msg.header = header
                self.csrt_pub.publish(csrt_msg)
                return
            
            # draw_image用データ準備
            xyxy = []
            for bbox in csrt_results:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                xyxy.append([x, y, x + w, y + h])
            
            # RGB変換してdraw_image実行
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            empty_masks = np.zeros((len(xyxy), image.shape[0], image.shape[1]), dtype=bool)
            probs = np.ones(len(xyxy))  # CSRTは信頼度1.0
            
            csrt_image_rgb = draw_image(image_rgb, empty_masks, np.array(xyxy), probs, tracker_labels)
            
            # BGR変換してROS配信
            csrt_image_bgr = cv2.cvtColor(csrt_image_rgb, cv2.COLOR_RGB2BGR)
            csrt_msg = self.cv_bridge.cv2_to_imgmsg(csrt_image_bgr, 'bgr8')
            csrt_msg.header = header
            self.csrt_pub.publish(csrt_msg)
            
        except Exception as e:
            self.logger.error(f"CSRT可視化エラー: {e}")
    
    def _process_sam_segmentation(self, csrt_results: list, tracker_labels: list,
                                image: np.ndarray, header):
        """SAM2セグメンテーション処理（毎フレーム実行・エラーハンドリング強化）"""
        try:
            if not csrt_results:
                return
            
            # トラッキング結果をSAM2形式に変換
            boxes = [[x, y, x+w, y+h] for x, y, w, h in csrt_results]
            
            # SAM2推論（ModelCoordinator経由）
            from PIL import Image as PILImage
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # SAM2セグメンテーション実行（エラーハンドリング強化）
            try:
                sam_result = self.tracker.coordinator.sam.predict_batch([image_rgb], xyxy=[boxes])
                
                # SAM結果の安全な取得（len()エラー対策）
                if sam_result is not None:
                    try:
                        # sam_resultがリストまたはタプルかどうかを先にチェック
                        if hasattr(sam_result, '__len__') and hasattr(sam_result, '__getitem__'):
                            if len(sam_result) >= 2:
                                masks, mask_scores = sam_result[0], sam_result[1]
                            else:
                                masks, mask_scores = None, None
                        else:
                            # sam_resultがスカラーの場合
                            masks, mask_scores = None, None
                    except (TypeError, AttributeError) as e:
                        self.logger.debug(f"SAM結果解析エラー: {e}")
                        masks, mask_scores = None, None
                else:
                    masks, mask_scores = None, None
                    
            except Exception as sam_error:
                self.logger.warning(f"SAM2推論エラー: {sam_error}")
                masks, mask_scores = None, None
            
            if masks is not None:
                try:
                    # masksの安全なチェック
                    has_masks = False
                    if hasattr(masks, '__len__') and hasattr(masks, 'size'):
                        has_masks = getattr(masks, 'size', 0) > 0 or (hasattr(masks, '__len__') and len(masks) > 0)
                    elif hasattr(masks, '__len__'):
                        has_masks = len(masks) > 0
                    elif masks is not None:
                        has_masks = True
                    
                    if has_masks:
                        result_masks = masks[0] if isinstance(masks, (list, tuple)) and len(masks) > 0 else masks
                        
                        # draw_imageを使用して可視化
                        xyxy = np.array(boxes)
                        
                        # mask_scoresの安全なチェック
                        if mask_scores is not None:
                            try:
                                if hasattr(mask_scores, '__len__') and len(mask_scores) > 0:
                                    probs = mask_scores[0] if isinstance(mask_scores, (list, tuple)) else mask_scores
                                else:
                                    probs = np.ones(len(boxes))
                            except (TypeError, AttributeError):
                                probs = np.ones(len(boxes))
                        else:
                            probs = np.ones(len(boxes))
                        
                        # RGB画像でdraw_image実行
                        sam_image_rgb = draw_image(image_rgb, result_masks, xyxy, probs, tracker_labels)
                        
                        # BGRに変換してROS配信
                        sam_image_bgr = cv2.cvtColor(sam_image_rgb, cv2.COLOR_RGB2BGR)
                        sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_image_bgr, 'bgr8')
                        sam_msg.header = header
                        self.sam_pub.publish(sam_msg)
                        return
                        
                except (TypeError, AttributeError) as e:
                    self.logger.debug(f"masks処理エラー: {e}")
                    
            # マスクなしの場合はバウンディングボックスのみ
            self._publish_sam_bbox_only(csrt_results, tracker_labels, image, header)
            
        except Exception as e:
            self.logger.error(f"SAM2セグメンテーションエラー: {e}")
            # エラー時はバウンディングボックスのみ表示
            self._publish_sam_bbox_only(csrt_results, tracker_labels, image, header)
    
    def _publish_sam_bbox_only(self, csrt_results: list, tracker_labels: list, 
                              image: np.ndarray, header):
        """SAMスキップ時のバウンディングボックスのみ表示"""
        try:
            if not csrt_results or not tracker_labels:
                empty_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                empty_msg.header = header
                self.sam_pub.publish(empty_msg)
                return
                
            # バウンディングボックスのみで軽量可視化
            boxes = [[x, y, x+w, y+h] for x, y, w, h in csrt_results]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            empty_masks = np.zeros((len(boxes), image.shape[0], image.shape[1]), dtype=bool)
            xyxy = np.array(boxes)
            probs = np.ones(len(boxes))
            
            sam_image_rgb = draw_image(image_rgb, empty_masks, xyxy, probs, tracker_labels)
            sam_image_bgr = cv2.cvtColor(sam_image_rgb, cv2.COLOR_RGB2BGR)
            sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_image_bgr, 'bgr8')
            sam_msg.header = header
            self.sam_pub.publish(sam_msg)
            
        except Exception as e:
            self.logger.debug(f"SAMボックス表示エラー: {e}")
            empty_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            empty_msg.header = header
            self.sam_pub.publish(empty_msg)
    
    def _publish_empty_images(self, image: np.ndarray, header):
        """空画像配信（トラッカーなし時）"""
        try:
            # 元画像をそのまま配信
            empty_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            empty_msg.header = header
            
            self.csrt_pub.publish(empty_msg)
            self.sam_pub.publish(empty_msg)
            
        except Exception as e:
            self.logger.error(f"空画像配信エラー: {e}")
    
    def destroy_node(self):
        """ノード終了時の非同期処理クリーンアップ"""
        self.logger.info("LangSAMトラッカーノード終了処理開始")
        
        # 非同期検出処理マネージャー終了
        if hasattr(self, 'async_detection_manager'):
            self.async_detection_manager.shutdown()
            
        # CSRT復旧機能終了（統計ログ出力）
        if self.csrt_frame_manager:
            status = self.csrt_frame_manager.get_manager_status()
            self.logger.info(f"Python CSRT復旧機能統計: フレーム数{status['frame_count']}, 復旧モード{status['recovery_mode']}")
            
        # C++ CSRT復旧統計
        if self.enable_csrt_recovery and self.csrt_client.is_available():
            recovery_stats = self.csrt_client.get_recovery_stats()
            self.logger.info(f"C++ CSRT復旧統計: 復旧成功{recovery_stats['recovered']}, 失敗{recovery_stats['failed']}")
        
        # 親クラスの終了処理
        super().destroy_node()
        self.logger.info("LangSAMトラッカーノード終了処理完了")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LangSAMTrackerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
