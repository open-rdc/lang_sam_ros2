#!/usr/bin/env python3
"""
LangSAM Tracker Node with Native C++ CSRT Implementation (Synchronous Version)
ハイブリッドPython/C++実装によるリアルタイムオブジェクト追跡ノード
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Polygon, Point32
import time

from lang_sam.tracker_utils.lang_sam_tracker import LangSAMTracker
from lang_sam.models.utils import DEVICE
from lang_sam.tracker_utils.csrt_client import CSRTClient
from lang_sam.utils import draw_image
from lang_sam_wrapper.fast_processing_client import get_fast_processing_client

# カスタムメッセージ型をインポート
try:
    from lang_sam_msgs.msg import DetectionResult
    from geometry_msgs.msg import Polygon, Point32
    LANG_SAM_MSGS_AVAILABLE = True
    print("Successfully imported lang_sam_msgs.msg.DetectionResult")
except ImportError as e:
    LANG_SAM_MSGS_AVAILABLE = False
    print(f"Failed to import lang_sam_msgs: {e}")


class Detection:
    """検出結果クラス
    GroundingDINO検出結果をROS2メッセージ形式に変換する目的で使用
    """
    def __init__(self, box, label, score=1.0):
        self.x = int(box[0])
        self.y = int(box[1])
        self.width = int(box[2] - box[0])
        self.height = int(box[3] - box[1])
        self.label = label
        self.score = score


class LangSAMTrackerNode(Node):
    """Main LangSAM node with C++ CSRT tracker integration (Synchronous Version)
    
    技術的な処理フロー：
    1. GroundingDINO: テキストプロンプトベースのゼロショット物体検出（1Hz）
    2. C++ CSRT: 高速リアルタイム追跡（全フレーム処理）
    3. SAM2: セグメンテーションマスク生成（10Hz独立実行）
    """
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        print(f"[INIT] LANG_SAM_MSGS_AVAILABLE at __init__: {LANG_SAM_MSGS_AVAILABLE}")
        
        self.logger = self.get_logger()
        self.logger.info("LangSAMトラッカーノード（C++ CSRT版）同期処理で初期化開始")
        
        # ROS2標準パラメータ管理で設定を読み込み
        # config.yamlから27個のCSRTパラメータを含む全設定を管理する目的で使用
        self._declare_parameters()
        self._load_parameters()
        self.logger.info("パラメータ読み込み完了")
        
        # CV Bridge を初期化
        # ROS2画像メッセージとOpenCV画像形式の相互変換の目的で使用
        self.cv_bridge = CvBridge()
        
        # 高速処理クライアントを初期化
        # C++実装のBGR→RGB変換とキャッシュ機構により画像処理を高速化する目的で使用
        self.fast_client = get_fast_processing_client()
        self.logger.info("高速化処理クライアント初期化完了")
        
        # LangSAMトラッカーを初期化
        # ROS2パラメータから設定値を直接使用
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
            
        # ネイティブC++ CSRTクライアントを初期化
        # pybind11経由でC++実装のCSRTトラッカーを利用し、リアルタイム性能を確保する目的で使用
        self.logger.info("C++ CSRTクライアント初期化開始")
        self.csrt_client = CSRTClient(self)
        
        if not self.csrt_client.is_available():
            self.logger.error("ネイティブC++ CSRT拡張が利用できません、終了します")
            raise RuntimeError("C++ CSRT initialization failed")
            
        self.logger.info("ネイティブC++ CSRTクライアント初期化完了")
        
        # LangSAMトラッカーを初期化
        # GroundingDINOとSAM2モデルを統合管理し、GPU推論を制御する目的で使用
        self.logger.info("LangSAMトラッカー初期化開始")
        self.lang_sam_tracker = LangSAMTracker(
            sam_type=sam_model,  # SAM2モデルバリアント選択（tiny/small/base/large）
            device=str(DEVICE)   # CUDA対応GPU自動検出
        )
        self.logger.info("LangSAMトラッカー初期化完了")
        
        # 設定を保存
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.gdino_interval_seconds = gdino_interval_seconds
        self.tracking_targets = tracking_targets
        self.bbox_margin = bbox_margin
        self.bbox_min_size = bbox_min_size
        
        # SAM2独立実行用設定
        # CSRTトラッキングから独立して10Hzでセグメンテーションを実行する目的で使用
        self.sam2_interval_seconds = 0.1  # デフォルト10Hz
        self.sam2_independent_mode = True
        
        # タイミング制御
        self.last_gdino_time = 0.0
        self.last_sam2_time = 0.0  # SAM2独立タイマー
        self.frame_count = 0
        
        # CSRTトラッキング結果のキャッシュ（SAM2独立実行用）
        # 非同期でSAM2が最新のトラッキング結果を参照できるようにする目的で使用
        self.cached_csrt_results = []  # バウンディングボックス座標
        self.cached_csrt_labels = []   # オブジェクトラベル
        self.cached_image = None       # 最新フレーム画像
        
        # パブリッシャー
        self.gdino_pub = self.create_publisher(Image, gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, csrt_output_topic, 10)
        self.sam_pub = self.create_publisher(Image, sam_topic, 10)
        
        # Detection results publisher for navigation (always create regardless of lang_sam_msgs)
        print(f"\n=== PUBLISHER CREATION DEBUG ===")
        print(f"[PUBLISHER] Creating DetectionResult publisher, LANG_SAM_MSGS_AVAILABLE: {LANG_SAM_MSGS_AVAILABLE}")
        print(f"[PUBLISHER] DetectionResult class available: {DetectionResult if LANG_SAM_MSGS_AVAILABLE else 'NOT AVAILABLE'}")
        self.logger.info(f"LANG_SAM_MSGS_AVAILABLE: {LANG_SAM_MSGS_AVAILABLE}")
        
        if LANG_SAM_MSGS_AVAILABLE:
            try:
                self.detection_pub = self.create_publisher(DetectionResult, '/lang_sam_detections', 10)
                print("[PUBLISHER] ✓ DetectionResult publisher created successfully on /lang_sam_detections")
                self.logger.info("✓ DetectionResult publisher created on /lang_sam_detections")
                self.use_fallback_publisher = False
            except Exception as e:
                print(f"[PUBLISHER] ✗ Failed to create DetectionResult publisher: {e}")
                self.logger.error(f"Failed to create DetectionResult publisher: {e}")
                self.use_fallback_publisher = True
                # Fall through to fallback creation
        else:
            self.use_fallback_publisher = True
        
        if self.use_fallback_publisher:
            # Fallback: use standard ROS2 messages for compatibility
            print("[PUBLISHER] Creating fallback String publisher...")
            from sensor_msgs.msg import CompressedImage
            from std_msgs.msg import Header, String
            self.detection_pub = self.create_publisher(String, '/lang_sam_detections_simple', 10)
            print("[PUBLISHER] ✓ Fallback String publisher created on /lang_sam_detections_simple")
            self.logger.info("✓ Fallback String publisher created for lane following")
        
        print(f"=== PUBLISHER CREATION COMPLETE ===\n")
        
        # サブスクライバー
        self.logger.info(f"画像サブスクリプション作成開始: {input_topic}")
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        self.logger.info(f"画像サブスクリプション作成完了: {input_topic}")
        
        # 現在のトラッキング状態
        self.current_detection_labels = []
        
        # 配信用の検出結果キャッシュ
        self.last_detection_boxes = []
        self.last_detection_labels = []
        self.last_detection_masks = []
        self.last_detection_probs = []
        
        self.logger.info("LangSAMトラッカーノード初期化完了（同期処理版 + C++高速化）")
        
        # 高速化統計情報をログ出力
        cache_size = self.fast_client.get_cache_size()
        self.logger.info(f"高速化統計: キャッシュサイズ={cache_size}")
    
    def _declare_parameters(self):
        """Declare parameters (fallback)"""
        self.declare_parameter('sam_model', 'sam2.1_hiera_tiny')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.25)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('gdino_interval_seconds', 1.0)
        
        # トピックパラメータ
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_output_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # トラッキングパラメータ
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
        """同期画像処理コールバック
        
        処理タイミング制御：
        - GroundingDINO: gdino_interval_seconds間隔で実行（デフォルト1.0秒）
        - CSRT: 毎フレーム実行（リアルタイム追跡）  
        - SAM2: sam2_interval_seconds間隔で独立実行（デフォルト0.1秒）
        """
        self.logger.info(f"画像コールバック開始: フレーム{self.frame_count + 1}")
        
        try:
            # ROS画像をOpenCVへ変換
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            current_time = time.time()
            self.frame_count += 1
            
            # 画像をキャッシュ（SAM2独立実行用）
            self.cached_image = image.copy()
            
            # GroundingDINO検出（同期処理）
            should_run_gdino = (current_time - self.last_gdino_time) >= self.gdino_interval_seconds
            
            self.logger.debug(f"GDINO判定: current={current_time:.3f}, last={self.last_gdino_time:.3f}, interval={self.gdino_interval_seconds}, should_run={should_run_gdino}")
            
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
            
            # SAM2セグメンテーション（独立実行モード）
            if self.sam2_independent_mode:
                should_run_sam2 = (current_time - self.last_sam2_time) >= self.sam2_interval_seconds
                if should_run_sam2 and self.cached_csrt_results:
                    self.logger.debug(f"フレーム{self.frame_count}: SAM2独立実行 (10Hz)")
                    self._process_sam_segmentation_independent(image, msg.header)
                    self.last_sam2_time = current_time
        
        except Exception as e:
            self.logger.error(f"画像コールバックエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_grounding_dino_detection(self, image: np.ndarray) -> list:
        """GroundingDINO同期検出実行
        
        テキストプロンプトから物体を検出する目的で使用
        box_threshold/text_thresholdにより検出精度を制御
        """
        try:
            from PIL import Image as PILImage
            import cv2
            
            # C++高速BGR→RGB変換（キャッシュあり）
            # OpenCV形式（BGR）をPIL/AI モデル用（RGB）に変換する目的で使用
            rgb_image = self.fast_client.bgr_to_rgb_cached(image)
            pil_image = PILImage.fromarray(rgb_image)
            
            start_time = time.time()
            
            # GroundingDINO推論実行（元のPython実装で動作確認）
            results = self.lang_sam_tracker.predict_with_tracking(
                images_pil=[pil_image],
                texts_prompt=[self.text_prompt],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                update_trackers=False,
                run_sam=False
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
        """検出結果の適用処理
        
        GroundingDINO検出結果をC++ CSRTトラッカーに登録する目的で使用
        tracking_targetsリストでフィルタリング
        """
        # C++ CSRTトラッカー初期化
        detection_boxes = []
        detection_labels = []
        detection_boxes_for_msg = []  # For message publishing
        
        for det in detections:
            if hasattr(det, 'label') and det.label in self.tracking_targets:
                bbox = (det.x, det.y, det.width, det.height)
                detection_boxes.append(bbox)
                detection_labels.append(det.label)
                # Convert to [x1, y1, x2, y2] format for message
                detection_boxes_for_msg.append([det.x, det.y, det.x + det.width, det.y + det.height])
        
        if detection_boxes:
            # C++ CSRT初期化
            # 新規検出オブジェクトのトラッカーを作成し、既存トラッカーを更新する目的で使用
            csrt_results = self.csrt_client.process_detections(
                current_image, detection_boxes, detection_labels
            )
            self.logger.info(f"C++ CSRT初期化完了: {len(csrt_results)}トラッカー")
            
            # 検出結果ラベル保存
            self.current_detection_labels = detection_labels.copy()
            
            # Note: Detection results will be published after SAM2 processing completes with masks
    
    def _process_csrt_tracking(self, image: np.ndarray, header):
        """CSRT継続トラッキング処理
        
        毎フレーム実行されるリアルタイム追跡処理
        C++実装により高速化を実現する目的で使用
        """
        # CSRT更新
        csrt_results = self.csrt_client.update_trackers(image)
        
        if csrt_results:
            # CSRTラベル取得
            tracker_labels = self.csrt_client.get_cached_labels()
            
            # CSRT結果をキャッシュ（SAM2独立実行用）
            self.cached_csrt_results = csrt_results.copy()
            self.cached_csrt_labels = tracker_labels.copy()
            
            # CSRT可視化生成・配信
            self._publish_csrt_visualization(csrt_results, tracker_labels, image, header)
            
            # SAM2セグメンテーション（非独立モード時のみ）
            if not self.sam2_independent_mode:
                self._process_sam_segmentation(csrt_results, tracker_labels, image, header)
        else:
            # トラッカーなし
            self.cached_csrt_results = []
            self.cached_csrt_labels = []
            # 空画像配信
            self._publish_empty_images(image, header)
    
    def _publish_gdino_visualization(self, detections: list, image: np.ndarray, header):
        """GroundingDINO可視化配信（高速化版）"""
        try:
            if not detections:
                # 検出なしの場合は元画像をそのまま配信
                gdino_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                gdino_msg.header = header
                self.gdino_pub.publish(gdino_msg)
                return
            
            # draw_image用データ準備
            xyxy = []
            labels_list = []
            
            for detection in detections:
                x, y, w, h = detection.x, detection.y, detection.width, detection.height
                xyxy.append([x, y, x + w, y + h])
                labels_list.append(detection.label)
            
            # draw_imageで可視化
            from lang_sam.utils import draw_image
            if xyxy and labels_list:
                # probsを適切に生成
                probs = [1.0] * len(xyxy)
                gdino_img = draw_image(
                    image_rgb=image,
                    masks=[],    # 空のリストを渡す  
                    xyxy=xyxy,
                    probs=probs,     # 確率値を渡す
                    labels=labels_list
                )
            else:
                gdino_img = image.copy()  # 検出なしの場合は元画像
            
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
            from lang_sam.utils import draw_image
            # probsを適切に生成
            probs = [1.0] * len(xyxy)
            csrt_img = draw_image(
                image_rgb=image,
                masks=[],  # 空のリストを渡す
                xyxy=xyxy,
                probs=probs,   # 確率値を渡す
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
        """SAM2セグメンテーション実行 (draw_image使用)
        
        バウンディングボックスからピクセルレベルのセグメンテーションマスクを生成する目的で使用
        SAM2のゼロショット能力により事前学習なしで動作
        """
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
                    # Transformerベースのセグメンテーションモデルで高精度マスク生成する目的で使用
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
    
    def _run_sam2_segmentation_with_masks(self, image: np.ndarray, pil_image, boxes: list, labels: list):
        """SAM2セグメンテーション実行してマスクも返す"""
        try:
            # ModelCoordinatorのSAMにアクセス
            sam_predictor = self.lang_sam_tracker.coordinator.sam
            
            if sam_predictor is None:
                self.logger.warning("SAM2 predictor not available")
                return self._create_bbox_visualization(image, boxes, labels), []
            
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
                    # Transformerベースのセグメンテーションモデルで高精度マスク生成する目的で使用
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
                return sam_img, masks
            else:
                # セグメンテーション失敗時はボックスのみ表示
                return self._create_bbox_visualization(image, boxes, labels), []
                
        except Exception as e:
            self.logger.warning(f"SAM2セグメンテーション実行エラー: {e}")
            return self._create_bbox_visualization(image, boxes, labels), []
    
    def _process_sam_segmentation_independent(self, image: np.ndarray, header):
        """SAM2独立実行処理（CSRTトラッキング結果を使用）
        
        10Hz固定レートでセグメンテーションを実行する目的で使用
        CSRTの高速追跡とSAM2の精密セグメンテーションを分離して最適化
        """
        try:
            if not self.cached_csrt_results:
                # トラッキング結果なし、元画像配信
                sam_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                sam_msg.header = header
                self.sam_pub.publish(sam_msg)
                return
            
            # SAM2推論実行
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # キャッシュされたCSRTバウンディングボックスを使用
            boxes = []
            for bbox in self.cached_csrt_results:
                x, y, w, h = bbox
                boxes.append([x, y, x + w, y + h])
            
            if boxes:
                # SAM2セグメンテーション実行
                sam_img, masks = self._run_sam2_segmentation_with_masks(image, pil_image, boxes, self.cached_csrt_labels)
                
                # ROS配信
                sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_img, 'bgr8')
                sam_msg.header = header
                self.sam_pub.publish(sam_msg)
                
                # Update detection results with SAM2 masks
                if masks:
                    self.logger.info(f"SAM2生成マスク数: {len(masks)}")
                    for i, mask in enumerate(masks):
                        self.logger.debug(f"マスク{i}: shape={mask.shape}, dtype={mask.dtype}")
                    
                    self._cache_detection_results(
                        boxes=boxes,
                        labels=self.cached_csrt_labels,
                        masks=masks,
                        probs=[1.0] * len(boxes)
                    )
                    
                    # Publish updated detection results with masks
                    self._publish_detection_results(
                        boxes=boxes,
                        labels=self.cached_csrt_labels,
                        masks=masks,
                        probs=[1.0] * len(boxes),
                        header=header
                    )
                else:
                    self.logger.warn("SAM2でマスクが生成されませんでした")
                
                self.logger.debug(f"SAM2独立実行完了: {len(boxes)}オブジェクト, {len(masks) if masks else 0}マスク")
            else:
                # ボックスなし、元画像配信
                sam_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                sam_msg.header = header
                self.sam_pub.publish(sam_msg)
                
        except Exception as e:
            self.logger.warning(f"SAM2独立実行エラー: {e}")
            # エラー時は元画像を配信
            sam_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            sam_msg.header = header
            self.sam_pub.publish(sam_msg)
    
    def _publish_empty_images(self, image: np.ndarray, header):
        """トラッカーなし時の空画像配信"""
        # 元画像をそのまま全トピックに配信
        image_msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
        image_msg.header = header
        
        self.csrt_pub.publish(image_msg)
        # SAM画像（非独立モード時のみ）
        if not self.sam2_independent_mode:
            self.sam_pub.publish(image_msg)
    
    def _publish_detection_results(self, boxes: list, labels: list, masks: list, probs: list, header):
        """検出結果をDetectionResultメッセージで配信
        
        ナビゲーションノードにセグメンテーション情報を提供する目的で使用
        カスタムメッセージ型でボックス・マスク・ラベルを統合配信
        """
        print(f"[PUBLISH] _publish_detection_results called with {len(boxes)} boxes, {len(labels)} labels")
        print(f"[PUBLISH] LANG_SAM_MSGS_AVAILABLE: {LANG_SAM_MSGS_AVAILABLE}")
        print(f"[PUBLISH] use_fallback_publisher: {self.use_fallback_publisher}")
        print(f"[PUBLISH] detection_pub is None: {self.detection_pub is None}")
        
        if self.use_fallback_publisher:
            print("[PUBLISH] Using fallback String publisher")
            self.logger.warn("Using fallback String publisher")
            self._publish_detection_results_fallback(boxes, labels, masks, probs, header)
            return
        if self.detection_pub is None:
            print("[PUBLISH] Detection publisher is None, skipping")
            self.logger.warn("Detection publisher is None, skipping detection publish")
            return
        
        print("[PUBLISH] Using DetectionResult publisher")
            
        try:
            # Create DetectionResult message
            detection_msg = DetectionResult()
            detection_msg.header = header
            detection_msg.num_detections = len(boxes)
            detection_msg.model_used = "GroundingDINO+SAM2+CSRT"
            
            # Convert boxes to Polygon format
            for box in boxes:
                polygon = Polygon()
                if len(box) == 4:  # [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = box
                    # Create rectangle polygon
                    polygon.points = [
                        Point32(x=float(x1), y=float(y1), z=0.0),
                        Point32(x=float(x2), y=float(y1), z=0.0),
                        Point32(x=float(x2), y=float(y2), z=0.0),
                        Point32(x=float(x1), y=float(y2), z=0.0)
                    ]
                detection_msg.boxes.append(polygon)
            
            # Add labels and probabilities
            detection_msg.labels = labels
            detection_msg.probabilities = [float(p) for p in probs]
            
            # Convert masks to sensor_msgs/Image format
            self.logger.debug(f"マスク変換開始: {len(masks)}個のマスクを処理")
            for i, mask in enumerate(masks):
                if mask is not None:
                    try:
                        self.logger.debug(f"マスク{i}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}")
                        
                        # Ensure mask is 2D boolean or uint8
                        if mask.dtype == bool:
                            mask = mask.astype(np.uint8) * 255
                        elif mask.dtype != np.uint8:
                            mask = (mask * 255).astype(np.uint8)
                        
                        # Convert to grayscale if needed
                        if len(mask.shape) == 3:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        
                        # Ensure mask is contiguous in memory
                        mask = np.ascontiguousarray(mask)
                        
                        # Convert to ROS Image message
                        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask, encoding="mono8")
                        mask_msg.header = header
                        detection_msg.masks.append(mask_msg)
                        self.logger.debug(f"マスク{i}変換成功: {mask_msg.width}x{mask_msg.height}")
                    except Exception as e:
                        self.logger.error(f"マスク{i}変換エラー: {e}")
                        # Create empty mask as fallback
                        empty_mask = np.zeros((100, 100), dtype=np.uint8)
                        mask_msg = self.cv_bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
                        mask_msg.header = header
                        detection_msg.masks.append(mask_msg)
                else:
                    self.logger.debug(f"マスク{i}はNoneです - 空のマスクを作成")
                    # Create empty mask for None values
                    empty_mask = np.zeros((100, 100), dtype=np.uint8)
                    mask_msg = self.cv_bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
                    mask_msg.header = header
                    detection_msg.masks.append(mask_msg)
            
            # Publish the detection result
            print(f"[PUBLISH] About to publish DetectionResult message")
            try:
                self.detection_pub.publish(detection_msg)
                print(f"[PUBLISH] ✓ DetectionResult message published successfully")
                self.logger.info(f"検出結果配信完了: {len(boxes)}ボックス, {len(detection_msg.masks)}マスク, {len(labels)}ラベル")
            except Exception as publish_e:
                print(f"[PUBLISH] ✗ Failed to publish DetectionResult: {publish_e}")
                self.logger.error(f"Failed to publish DetectionResult: {publish_e}")
                raise
            
        except Exception as e:
            self.logger.error(f"検出結果配信エラー: {e}")
    
    def _publish_detection_results_fallback(self, boxes: list, labels: list, masks: list, probs: list, header):
        """Fallback: 検出結果をStringメッセージで配信"""
        print(f"[FALLBACK] Starting fallback detection publishing with {len(boxes)} boxes, {len(labels)} labels")
        self.logger.info(f"[FALLBACK] Publishing detection results via String message: {len(boxes)} detections")
        
        try:
            import json
            
            # Create detection data structure
            detection_data = {
                "timestamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
                "frame_id": header.frame_id,
                "num_detections": len(boxes),
                "model_used": "GroundingDINO+SAM2+CSRT",
                "boxes": [],
                "labels": labels,
                "probabilities": [float(p) for p in probs],
                "has_masks": len(masks) > 0 and any(mask is not None for mask in masks)
            }
            
            # Convert boxes to simple format [x1, y1, x2, y2]
            for box in boxes:
                if len(box) == 4:
                    detection_data["boxes"].append([float(x) for x in box])
            
            # Add mask information (simplified - just indicate if white line masks exist)
            white_line_masks = []
            for i, label in enumerate(labels):
                if label == "white line" and i < len(masks) and masks[i] is not None:
                    mask = masks[i]
                    # Convert mask to binary and find contours for simplified representation
                    if mask.dtype == bool:
                        mask_binary = mask.astype(np.uint8) * 255
                    else:
                        mask_binary = (mask > 0).astype(np.uint8) * 255
                    
                    # Store mask dimensions and non-zero pixel count as proxy
                    white_line_masks.append({
                        "index": i,
                        "shape": mask.shape,
                        "non_zero_pixels": int(np.count_nonzero(mask_binary))
                    })
            
            detection_data["white_line_masks"] = white_line_masks
            
            # Publish as JSON string
            json_str = json.dumps(detection_data)
            string_msg = String()
            string_msg.data = json_str
            
            print(f"[FALLBACK] Publishing JSON string with {len(json_str)} characters")
            print(f"[FALLBACK] JSON preview: {json_str[:200]}...")
            self.detection_pub.publish(string_msg)
            print(f"[FALLBACK] ✓ Message published successfully")
            self.logger.info(f"[FALLBACK] 検出結果配信完了: {len(boxes)}ボックス, {len(white_line_masks)}白線マスク, {len(labels)}ラベル")
            
        except Exception as e:
            self.logger.error(f"[FALLBACK] 検出結果配信エラー: {e}")
    
    def _cache_detection_results(self, boxes: list, labels: list, masks: list = None, probs: list = None):
        """検出結果をキャッシュ"""
        self.last_detection_boxes = boxes.copy() if boxes else []
        self.last_detection_labels = labels.copy() if labels else []
        self.last_detection_masks = masks.copy() if masks else []
        self.last_detection_probs = probs.copy() if probs else [1.0] * len(boxes)
    
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