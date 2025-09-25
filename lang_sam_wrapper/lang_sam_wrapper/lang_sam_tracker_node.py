#!/usr/bin/env python3
"""
LangSAM Tracker Node - ROS2統合
OpticalFlowトラッキングとSAM2セグメンテーションのROS2ノード
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from typing import Dict, Any, Optional, List
from functools import lru_cache

from lang_sam.lang_sam_tracker import LangSamTracker
from lang_sam.models.utils import DEVICE
from PIL import Image as PILImage

# カスタムメッセージ型をインポート
try:
    from lang_sam_msgs.msg import DetectionResult
    from geometry_msgs.msg import Polygon, Point32
    LANG_SAM_MSGS_AVAILABLE = True
except ImportError:
    LANG_SAM_MSGS_AVAILABLE = False
    from std_msgs.msg import String


class LangSAMTrackerNode(Node):
    """最適化されたLangSAMトラッカーノード
    
    統合処理メソッドを使用してコード量を大幅削減
    """
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node')
        
        self.logger = self.get_logger()
        self.logger.info("LangSAMトラッカーノード初期化開始")
        
        # パラメータ宣言・読み込み
        self._setup_parameters()
        
        # CV Bridge初期化
        self.cv_bridge = CvBridge()
        
        # 最適化: メモリ事前割り当て
        self._image_rgb_cache: Optional[np.ndarray] = None
        self._pil_image_cache: Optional[PILImage.Image] = None
        self._result_cache: Dict[str, Any] = {}
        self._frame_skip_counter = 0
        
        # パフォーマンスカウンター
        self._total_processing_time = 0.0
        self._gdino_processing_time = 0.0
        self._gdino_execution_count = 0
        
        # LangSAMトラッカー初期化（OpticalFlow統合処理対応）
        self.lang_sam_tracker = LangSamTracker(
            sam_model=self.sam_model,
            device=str(DEVICE),
            # OpticalFlowパラメータを設定ファイルから渡す
            optical_flow_max_corners=self.optical_flow_max_corners,
            optical_flow_quality_level=self.optical_flow_quality_level,
            optical_flow_min_distance=self.optical_flow_min_distance,
            optical_flow_block_size=self.optical_flow_block_size,
            optical_flow_win_size=(self.optical_flow_win_size_x, self.optical_flow_win_size_y),
            optical_flow_max_level=self.optical_flow_max_level,
            optical_flow_max_disappeared=self.optical_flow_max_disappeared,
            optical_flow_min_tracked_points=self.optical_flow_min_tracked_points,
            # 適応的BBOXパラメータを設定ファイルから渡す
            enable_adaptive_bbox=self.enable_adaptive_bbox,
            bbox_scale_factor=self.bbox_scale_factor,
            min_bbox_scale=self.min_bbox_scale,
            max_bbox_scale=self.max_bbox_scale
        )
        
        # タイミング制御（正確な初期化）
        current_init_time = time.time()
        # 初回実行を促すため、過去の時刻を設定
        # 目的: ノード起動直後にGroundingDINOを実行する目的で使用
        self.last_gdino_time = current_init_time - self.gdino_interval_seconds - 1.0
        self.last_sam2_time = current_init_time - self.sam2_interval_seconds - 1.0
        self.frame_count = 0
        
        
        # パブリッシャー初期化
        self.gdino_pub = self.create_publisher(Image, self.gdino_topic, 10)
        self.optical_flow_pub = self.create_publisher(Image, self.optical_flow_output_topic, 10)
        self.sam_pub = self.create_publisher(Image, self.sam_topic, 10)
        
        # 検出結果パブリッシャー
        if LANG_SAM_MSGS_AVAILABLE:
            self.detection_pub = self.create_publisher(DetectionResult, '/lang_sam_detections', 10)
            self.use_fallback = False
        else:
            self.detection_pub = self.create_publisher(String, '/lang_sam_detections_simple', 10)
            self.use_fallback = True
        
        # サブスクライバー初期化
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )
        
        self.logger.info("LangSAMトラッカーノード初期化完了")
    
    def _setup_parameters(self):
        """パラメータ設定（統合版）"""
        # AIモデルパラメータ
        self.declare_parameter('sam_model', 'sam2.1_hiera_tiny')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.25)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('gdino_interval_seconds', 1.0)
        self.declare_parameter('sam2_interval_seconds', 0.1)
        
        # OpticalFlowパラメータ
        self.declare_parameter('optical_flow_max_corners', 100)
        self.declare_parameter('optical_flow_quality_level', 0.01)
        self.declare_parameter('optical_flow_min_distance', 10)
        self.declare_parameter('optical_flow_block_size', 7)
        self.declare_parameter('optical_flow_win_size_x', 15)
        self.declare_parameter('optical_flow_win_size_y', 15)
        self.declare_parameter('optical_flow_max_level', 2)
        self.declare_parameter('optical_flow_max_disappeared', 30)
        self.declare_parameter('optical_flow_min_tracked_points', 5)

        # 適応的BBOXパラメータ
        self.declare_parameter('enable_adaptive_bbox', True)
        self.declare_parameter('bbox_scale_factor', 0.02)
        self.declare_parameter('min_bbox_scale', 1.0)
        self.declare_parameter('max_bbox_scale', 1.5)
        
        # トピックパラメータ  
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('optical_flow_output_topic', '/image_optical_flow')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # 追跡対象パラメータ
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # パラメータ取得（一括処理で高速化）
        params = {
            'sam_model': 'sam2.1_hiera_tiny',
            'text_prompt': 'white line. red pylon. human. car.',
            'box_threshold': 0.25,
            'text_threshold': 0.25,
            'gdino_interval_seconds': 1.0,
            'sam2_interval_seconds': 0.1
        }
        
        for key, default in params.items():
            setattr(self, key, self.get_parameter(key).value)
        
        # OpticalFlowパラメータ読み込み（一括処理）
        optical_params = {
            'optical_flow_max_corners': 100,
            'optical_flow_quality_level': 0.01,
            'optical_flow_min_distance': 10,
            'optical_flow_block_size': 7,
            'optical_flow_win_size_x': 15,
            'optical_flow_win_size_y': 15,
            'optical_flow_max_level': 2,
            'optical_flow_max_disappeared': 30,
            'optical_flow_min_tracked_points': 5
        }

        for key, default in optical_params.items():
            setattr(self, key, self.get_parameter(key).value)

        # 適応的BBOXパラメータ読み込み
        adaptive_bbox_params = {
            'enable_adaptive_bbox': True,
            'bbox_scale_factor': 0.02,
            'min_bbox_scale': 1.0,
            'max_bbox_scale': 1.5
        }

        for key, default in adaptive_bbox_params.items():
            setattr(self, key, self.get_parameter(key).value)
        
        # OpticalFlowパラメータ情報をログ出力
        # 目的: OpticalFlowトラッキングパラメータを確認可能にする目的で使用
        self.logger.info(f"OpticalFlow設定: max_corners={self.optical_flow_max_corners}, quality_level={self.optical_flow_quality_level}")

        # 適応的BBOXパラメータ情報をログ出力
        self.logger.info(f"適応的BBOX設定: 有効={self.enable_adaptive_bbox}, scale_factor={self.bbox_scale_factor}")
        self.logger.info(f"BBOXスケール範囲: {self.min_bbox_scale} - {self.max_bbox_scale}")

        # GDINO実行間隔の確認
        self.logger.info(f"GDINO実行間隔: {self.gdino_interval_seconds}秒")
        self.logger.info(f"SAM2実行間隔: {self.sam2_interval_seconds}秒")
        
        # トピックパラメータ（一括処理）
        topic_params = {
            'input_topic': '/zed/zed_node/rgb/image_rect_color',
            'gdino_topic': '/image_gdino',
            'optical_flow_output_topic': '/image_optical_flow',
            'sam_topic': '/image_sam'
        }
        
        for key, default in topic_params.items():
            setattr(self, key, self.get_parameter(key).value)
        
        self.tracking_targets = self.get_parameter('tracking_targets').value
    
    def image_callback(self, msg: Image):
        """統合画像処理コールバック"""
        start_time = time.time()
        try:
            # フレームスキップ判定
            self._frame_skip_counter += 1
            if self._frame_skip_counter % 2 == 0 and self.frame_count > 100:
                return
            
            # ROS2→OpenCV変換（高速化）
            image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            if image_bgr is None or image_bgr.size == 0:
                return
            
            # BGR→RGB変換（メモリ効率化）
            if self._image_rgb_cache is None or self._image_rgb_cache.shape != image_bgr.shape:
                self._image_rgb_cache = np.empty_like(image_bgr)
            
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB, dst=self._image_rgb_cache)
            image_rgb = self._image_rgb_cache
            
            # PIL変換（必要時のみ）
            image_pil = None
            
            # フレーム管理とタイミング判定（最適化）
            self.frame_count += 1
            current_time = time.time()
            gdino_elapsed = current_time - self.last_gdino_time
            sam2_elapsed = current_time - self.last_sam2_time
            should_run_gdino = gdino_elapsed >= self.gdino_interval_seconds
            should_run_sam2 = sam2_elapsed >= self.sam2_interval_seconds
            
            # デバッグ出力の最適化（ログ出力頻度削減）
            if self.frame_count <= 5:  # 最初の5フレームのみ詳細出力
                self.logger.info(f"フレーム{self.frame_count}: GDINO経過={gdino_elapsed:.2f}秒, should_run={should_run_gdino}")
            elif self.frame_count % 60 == 1:  # 60フレームごとに出力（2秒に1回）
                self.logger.info(f"パフォーマンス: GDINO平均={self._gdino_processing_time/max(1, self._gdino_execution_count):.3f}秒")
            
            # GroundingDINO処理
            gdino_result = None
            if should_run_gdino:
                # PIL変換（必要時のみ）
                if image_pil is None:
                    image_pil = PILImage.fromarray(image_rgb)
                
                gdino_start_time = time.time()
                self.last_gdino_time = current_time
                
                gdino_results = self.lang_sam_tracker.predict_gdino(
                    [image_pil], [self.text_prompt], 
                    self.box_threshold, self.text_threshold
                )
                
                if gdino_results and len(gdino_results) > 0:
                    gdino_result = gdino_results[0]
                    processing_time = time.time() - gdino_start_time
                    
                    # パフォーマンス統計更新
                    self._gdino_processing_time += processing_time
                    self._gdino_execution_count += 1
                    
                    if self.frame_count <= 10 or self._gdino_execution_count % 10 == 1:
                        self.logger.info(f"GDINO完了: {processing_time:.3f}秒, {len(gdino_result.get('boxes', []))}個検出")
            
            # 統合フレーム処理
            reset_tracker = should_run_gdino
            result = self.lang_sam_tracker.process_frame(image_rgb, gdino_result, reset_tracker)
            
            # 結果配信
            self._publish_results_optimized(gdino_result, result, image_rgb, image_pil, msg.header, bool(gdino_result), should_run_sam2)
            
            if should_run_sam2:
                self.last_sam2_time = current_time
            
            # パフォーマンス統計更新
            total_time = time.time() - start_time
            self._total_processing_time += total_time
            
            if self.frame_count % 100 == 0:  # 100フレームごとに統計出力
                avg_total = self._total_processing_time / self.frame_count
                self.logger.info(f"統計 - フレーム{self.frame_count}: 平均処理時間={avg_total*1000:.1f}ms")
                
        except Exception as e:
            self.logger.error(f"画像処理エラー: {e}")
            import traceback
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
    
    def _publish_results_optimized(self, gdino_result: dict, tracking_result: dict, image_rgb: np.ndarray, image_pil, header, run_gdino: bool, run_sam2: bool):
        """最適化された結果配信メソッド"""
        try:
            # GroundingDINO結果配信（必要時のみ）
            if run_gdino and gdino_result:
                self._publish_gdino_result(gdino_result, image_rgb, header)

            # OpticalFlowトラッキング結果配信
            self._publish_tracking_result(tracking_result, image_rgb, header)

            # SAM2結果配信とDetectionResult配信
            if run_sam2 and tracking_result.get("boxes"):
                # PIL変換（必要時のみ）
                if image_pil is None:
                    image_pil = PILImage.fromarray(image_rgb)
                # SAM2マスク付き結果を取得して配信
                sam_result_with_masks = self._publish_sam2_result_and_get_masks(tracking_result, image_rgb, image_pil, header)
                # マスク付き検出結果を配信
                if sam_result_with_masks and sam_result_with_masks.get("boxes"):
                    self._publish_detection_data(sam_result_with_masks, header)
            else:
                # SAM2なしの場合は通常の検出結果を配信
                if tracking_result["boxes"]:
                    self._publish_detection_data(tracking_result, header)

        except Exception as e:
            self.logger.error(f"結果配信エラー: {e}")
    
    def _publish_gdino_result(self, gdino_result: dict, image_rgb: np.ndarray, header):
        """GroundingDINO結果配信"""
        # テンソル変換の最適化
        boxes_tensor = gdino_result.get("boxes", [])
        scores_tensor = gdino_result.get("scores", [])
        
        if hasattr(boxes_tensor, 'cpu'):
            boxes_list = boxes_tensor.cpu().numpy().tolist()
            scores_list = scores_tensor.cpu().numpy().tolist()
        else:
            boxes_list = boxes_tensor.tolist() if hasattr(boxes_tensor, 'tolist') else boxes_tensor
            scores_list = scores_tensor.tolist() if hasattr(scores_tensor, 'tolist') else scores_tensor
        
        gdino_vis_result = {
            "boxes": boxes_list,
            "labels": gdino_result.get("labels", []),
            "scores": scores_list,
            "masks": [],
            "track_ids": []
        }
        
        gdino_vis = self.lang_sam_tracker.visualize(image_rgb, gdino_vis_result)
        gdino_msg = self.cv_bridge.cv2_to_imgmsg(gdino_vis, encoding='rgb8')
        gdino_msg.header = header
        self.gdino_pub.publish(gdino_msg)
    
    def _publish_tracking_result(self, tracking_result: dict, image_rgb: np.ndarray, header):
        """OpticalFlowトラッキング結果配信"""
        tracking_vis_result = {
            "boxes": tracking_result.get("boxes", []),
            "labels": tracking_result.get("labels", []),
            "scores": tracking_result.get("scores", []),
            "masks": [],
            "track_ids": []  # IDを表示しない
        }
        
        # OpticalFlow可視化（特徴点と移動ベクトル表示）
        optical_flow_vis = self.lang_sam_tracker.visualize_optical_flow(image_rgb)
        optical_flow_msg = self.cv_bridge.cv2_to_imgmsg(optical_flow_vis, encoding='rgb8')
        optical_flow_msg.header = header
        self.optical_flow_pub.publish(optical_flow_msg)
    
    def _publish_sam2_result_and_get_masks(self, tracking_result: dict, image_rgb: np.ndarray, image_pil, header):
        """SAM2結果配信とマスク付き結果を返す"""
        try:
            sam_masks = []
            image_np = np.array(image_pil)

            for box in tracking_result["boxes"]:
                try:
                    mask, _, _ = self.lang_sam_tracker.sam.predict(image_np, np.array(box))
                    sam_masks.append(mask[0] if mask is not None and len(mask) > 0 else None)
                except Exception as e:
                    sam_masks.append(None)

            sam_vis_result = {
                "boxes": tracking_result.get("boxes", []),
                "labels": tracking_result.get("labels", []),
                "scores": tracking_result.get("scores", []),
                "masks": sam_masks,
                "track_ids": []
            }

            # 可視化画像を配信
            sam_vis = self.lang_sam_tracker.visualize(image_rgb, sam_vis_result)
            sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_vis, encoding='rgb8')
            sam_msg.header = header
            self.sam_pub.publish(sam_msg)

            # マスク付き結果を返す（DetectionResult用）
            return sam_vis_result

        except Exception as e:
            self.logger.error(f"SAM2処理エラー: {e}")
            return None
    
    def _publish_results(self, gdino_result: dict, centroid_result: dict, image_rgb: np.ndarray, image_pil, header, run_gdino: bool, run_sam2: bool):
        """分離結果配信（GroundingDINO検出結果とCentroidトラッキング結果を完全分離）"""
        try:
            # GroundingDINO結果（検出時のみ、純粋な検出結果）
            if run_gdino and gdino_result:
                # 目的: GroundingDINOの純粋な検出結果のみを可視化する目的で使用
                # テンソルをCPUに移動してnumpy配列に変換
                # 目的: PyTorchテンソルをnumpy配列に変換してTensorのBoolean判定エラーを回避する目的で使用
                boxes_tensor = gdino_result.get("boxes", [])
                scores_tensor = gdino_result.get("scores", [])
                
                boxes_list = boxes_tensor.cpu().numpy().tolist() if hasattr(boxes_tensor, 'cpu') else (boxes_tensor.tolist() if hasattr(boxes_tensor, 'tolist') else boxes_tensor)
                scores_list = scores_tensor.cpu().numpy().tolist() if hasattr(scores_tensor, 'cpu') else (scores_tensor.tolist() if hasattr(scores_tensor, 'tolist') else scores_tensor)
                
                gdino_only_result = {
                    "boxes": boxes_list,
                    "labels": gdino_result.get("labels", []),
                    "scores": scores_list,
                    "masks": [],  # マスクなし
                    "track_ids": []  # トラッキングIDなし
                }
                gdino_vis = self.lang_sam_tracker.visualize(image_rgb, gdino_only_result)
                gdino_msg = self.cv_bridge.cv2_to_imgmsg(gdino_vis, encoding='rgb8')
                gdino_msg.header = header
                self.gdino_pub.publish(gdino_msg)
            
            # Centroid結果（常時、GroundingDINOラベル付きボックス、IDなし）
            # 目的: Centroidトラッキング結果をIDなしで可視化する目的で使用
            centroid_only_result = {
                "boxes": centroid_result.get("boxes", []),
                "labels": centroid_result.get("labels", []),
                "scores": centroid_result.get("scores", []),
                "masks": [],  # マスクなし
                "track_ids": []  # IDを表示しない
            }
            # OpticalFlow可視化（特徴点と移動ベクトル表示）
            optical_flow_vis = self.lang_sam_tracker.visualize_optical_flow(image_rgb)
            optical_flow_msg = self.cv_bridge.cv2_to_imgmsg(optical_flow_vis, encoding='rgb8')
            optical_flow_msg.header = header
            self.optical_flow_pub.publish(optical_flow_msg)
            
            # SAM2結果（セグメンテーション実行時のみ、マスク付き、IDなし）
            if run_sam2 and centroid_result.get("boxes"):
                # 目的: Centroidトラッキング結果にSAM2セグメンテーションマスクを追加する目的で使用
                sam_masks = []
                image_np = np.array(image_pil)
                
                for box in centroid_result["boxes"]:
                    try:
                        # SAM2でセグメンテーション実行
                        mask, _, _ = self.lang_sam_tracker.sam.predict(image_np, np.array(box))
                        if mask is not None and len(mask) > 0:
                            sam_masks.append(mask[0])
                        else:
                            sam_masks.append(None)
                    except Exception as e:
                        print(f"[SAM2] セグメンテーションエラー: {e}")
                        sam_masks.append(None)
                
                sam_only_result = {
                    "boxes": centroid_result.get("boxes", []),
                    "labels": centroid_result.get("labels", []),
                    "scores": centroid_result.get("scores", []),
                    "masks": sam_masks,  # SAM2で生成されたマスク
                    "track_ids": []  # IDを表示しない
                }
                
                sam_vis = self.lang_sam_tracker.visualize(image_rgb, sam_only_result)
                sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_vis, encoding='rgb8')
                sam_msg.header = header
                self.sam_pub.publish(sam_msg)
            
            # 検出結果配信（ナビゲーション用）
            if centroid_result["boxes"]:
                # SAM2実行時はマスク付き結果を配信
                if run_sam2 and 'sam_only_result' in locals():
                    self._publish_detection_data(sam_only_result, header)
                else:
                    self._publish_detection_data(centroid_result, header)
                
        except Exception as e:
            import traceback
            self.logger.error(f"結果配信エラー: {e}")
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
    
    @lru_cache(maxsize=32)
    def _get_cached_mask_msg(self, mask_hash: int, header_stamp: float):
        """マスクメッセージのキャッシュ（性能向上）"""
        return None  # 実装は必要に応じて
    
    def _publish_detection_data(self, result: dict, header):
        """検出データ配信（簡素化版）"""
        try:
            if self.use_fallback:
                # JSON形式で配信
                import json
                detection_data = {
                    "timestamp": header.stamp.sec + header.stamp.nanosec * 1e-9,
                    "boxes": result["boxes"],
                    "labels": result["labels"],
                    "track_ids": result.get("track_ids", []),
                    "num_detections": len(result["boxes"])
                }
                
                string_msg = String()
                string_msg.data = json.dumps(detection_data)
                self.detection_pub.publish(string_msg)
            else:
                # DetectionResult形式で配信
                detection_msg = DetectionResult()
                detection_msg.header = header
                detection_msg.num_detections = len(result["boxes"])
                detection_msg.labels = result["labels"]
                detection_msg.probabilities = result.get("scores", [1.0] * len(result["boxes"]))
                
                # ボックスをPolygon形式に変換
                for box in result["boxes"]:
                    x1, y1, x2, y2 = box
                    polygon = Polygon()
                    polygon.points = [
                        Point32(x=float(x1), y=float(y1), z=0.0),
                        Point32(x=float(x2), y=float(y1), z=0.0),
                        Point32(x=float(x2), y=float(y2), z=0.0),
                        Point32(x=float(x1), y=float(y2), z=0.0)
                    ]
                    detection_msg.boxes.append(polygon)
                
                # マスクをImage形式に変換（エンコーディング修正）
                for mask in result.get("masks", []):
                    if mask is not None:
                        # 型とサイズを正規化
                        if mask.dtype == bool:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                        elif mask.dtype == np.float32 or mask.dtype == np.float64:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                        else:
                            mask_uint8 = mask.astype(np.uint8)
                        
                        # 2Dに変換（必要に応じて）
                        if len(mask_uint8.shape) > 2:
                            mask_uint8 = mask_uint8.squeeze()
                        
                        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                        mask_msg.header = header
                        detection_msg.masks.append(mask_msg)
                
                self.detection_pub.publish(detection_msg)
                
        except Exception as e:
            self.logger.error(f"検出データ配信エラー: {e}")
    


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LangSAMTrackerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()