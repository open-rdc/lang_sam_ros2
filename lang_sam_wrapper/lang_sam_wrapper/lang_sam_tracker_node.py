#!/usr/bin/env python3
"""
LangSAM Tracker Node - 完全最適化版
統合処理メソッドを使用したシンプルなROS2ノード実装
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

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
            optical_flow_min_tracked_points=self.optical_flow_min_tracked_points
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
        
        # トピックパラメータ  
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('optical_flow_output_topic', '/image_optical_flow')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # 追跡対象パラメータ
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # パラメータ取得
        self.sam_model = self.get_parameter('sam_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        self.gdino_interval_seconds = self.get_parameter('gdino_interval_seconds').value
        self.sam2_interval_seconds = self.get_parameter('sam2_interval_seconds').value
        
        # OpticalFlowパラメータ読み込み
        self.optical_flow_max_corners = self.get_parameter('optical_flow_max_corners').value
        self.optical_flow_quality_level = self.get_parameter('optical_flow_quality_level').value
        self.optical_flow_min_distance = self.get_parameter('optical_flow_min_distance').value
        self.optical_flow_block_size = self.get_parameter('optical_flow_block_size').value
        self.optical_flow_win_size_x = self.get_parameter('optical_flow_win_size_x').value
        self.optical_flow_win_size_y = self.get_parameter('optical_flow_win_size_y').value
        self.optical_flow_max_level = self.get_parameter('optical_flow_max_level').value
        self.optical_flow_max_disappeared = self.get_parameter('optical_flow_max_disappeared').value
        self.optical_flow_min_tracked_points = self.get_parameter('optical_flow_min_tracked_points').value
        
        # OpticalFlowパラメータ情報をログ出力
        # 目的: OpticalFlowトラッキングパラメータを確認可能にする目的で使用
        self.logger.info(f"OpticalFlow設定: max_corners={self.optical_flow_max_corners}, quality_level={self.optical_flow_quality_level}")
        
        # GDINO実行間隔の確認
        self.logger.info(f"GDINO実行間隔: {self.gdino_interval_seconds}秒")
        self.logger.info(f"SAM2実行間隔: {self.sam2_interval_seconds}秒")
        
        self.input_topic = self.get_parameter('input_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.optical_flow_output_topic = self.get_parameter('optical_flow_output_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
        
        self.tracking_targets = self.get_parameter('tracking_targets').value
    
    def image_callback(self, msg: Image):
        """統合画像処理コールバック - 同期GroundingDINO版"""
        try:
            # ROS2→OpenCV変換
            image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            if image_bgr is None or image_bgr.size == 0:
                return
            
            # BGR→RGB変換
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = PILImage.fromarray(image_rgb)
            
            # フレーム管理とタイミング判定
            self.frame_count += 1
            current_time = time.time()
            gdino_elapsed = current_time - self.last_gdino_time
            sam2_elapsed = current_time - self.last_sam2_time
            should_run_gdino = gdino_elapsed >= self.gdino_interval_seconds
            should_run_sam2 = sam2_elapsed >= self.sam2_interval_seconds
            
            # デバッグ: タイミング情報を毎フレーム出力（最初の10フレーム）
            if self.frame_count <= 10:
                self.logger.info(f"フレーム{self.frame_count}: GDINO経過={gdino_elapsed:.2f}秒 (閾値={self.gdino_interval_seconds}), should_run={should_run_gdino}")
            elif self.frame_count % 30 == 1:  # 30フレームごとに出力
                self.logger.info(f"タイミング情報: GDINO経過={gdino_elapsed:.2f}秒, SAM2経過={sam2_elapsed:.2f}秒")
            
            # 同期GroundingDINO処理
            # 目的: シンプルな同期処理でフレーム同期を確保する目的で使用
            gdino_result = None
            if should_run_gdino:
                self.logger.info(f"GDINO実行開始: 前回から{current_time - self.last_gdino_time:.2f}秒経過")
                self.last_gdino_time = current_time
                
                start_time = time.time()
                gdino_results = self.lang_sam_tracker.predict_gdino(
                    [image_pil], [self.text_prompt], 
                    self.box_threshold, self.text_threshold
                )
                
                if gdino_results and len(gdino_results) > 0:
                    gdino_result = gdino_results[0]
                    processing_time = time.time() - start_time
                    self.logger.info(f"GDINO処理完了: {processing_time:.3f}秒")
                    self.logger.info(f"フレーム{self.frame_count}: {len(gdino_result.get('boxes', []))}オブジェクト検出")
            
            # リセットは行わない - 通常のトラッキングでマッチング
            # 目的: カルマンフィルタの速度情報を保持してBBOXの動きを実現する目的で使用
            # if gdino_result and len(gdino_result.get('boxes', [])) > 0:
            #     self.lang_sam_tracker.clear_trackers()  # リセットを廃止
                
            # Centroidトラッキング処理
            # 目的: GroundingDINO検出結果またはカルマンフィルタ予測でトラッキングする目的で使用
            boxes_tensor = gdino_result.get('boxes', []) if gdino_result else []
            has_detections = gdino_result is not None and len(boxes_tensor) > 0
            print(f"[DEBUG] GDINO結果確認: gdino_result={gdino_result is not None}, boxes={boxes_tensor}")
            
            if has_detections:
                # GroundingDINO検出時はトラッカーをリセット
                # 目的: BBOX更新時に古いトラッキング情報を破棄して蓄積を防ぐ目的で使用
                if should_run_gdino:  # GroundingDINOが実際に実行された場合のみリセット
                    self.lang_sam_tracker.clear_trackers()
                    print("[DEBUG] GroundingDINO新規検出によりトラッカーをリセット")
                
                # GroundingDINO結果をCentroidに渡す
                detections = []
                # テンソルをCPUに移動してnumpy配列に変換
                # 目的: PyTorchテンソルをnumpy配列に変換してTensorのBoolean判定エラーを回避する目的で使用
                boxes_np = boxes_tensor.cpu().numpy() if hasattr(boxes_tensor, 'cpu') else np.array(boxes_tensor)
                scores_tensor = gdino_result.get("scores", [])
                scores_np = scores_tensor.cpu().numpy() if hasattr(scores_tensor, 'cpu') else np.array(scores_tensor)
                
                for i, box in enumerate(boxes_np):
                    # Centroid要求形式: [x1, y1, x2, y2, confidence]
                    score = float(scores_np[i]) if i < len(scores_np) else 1.0
                    detections.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), score])
                
                print(f"[DEBUG] OpticalFlow入力detections: {len(detections)}個, detections={detections}")
                
                # OpticalFlow更新（GroundingDINOのラベルも一緒に渡す）
                # 目的: 検出結果とフレーム画像をOpticalFlowトラッカーに渡して更新する目的で使用
                track_result = self.lang_sam_tracker.track(image_rgb, detections, gdino_result.get("labels", []))
                print(f"[DEBUG] track_result keys: {track_result.keys()}")
                print(f"[DEBUG] track_result track_ids: {track_result.get('track_ids', [])} (type: {type(track_result.get('track_ids', []))})")
                
                # OpticalFlowは数値IDを直接使用したtrack結果を返す
                # 目的: OpticalFlow内部で数値IDが直接管理されているため単純に使用する目的で使用
                if track_result["boxes"] and track_result["track_ids"]:
                    tracks = np.array([[box[0], box[1], box[2], box[3], tid] for box, tid in zip(track_result["boxes"], track_result["track_ids"])], dtype=np.float32)
                else:
                    tracks = np.empty((0, 5), dtype=np.float32)
                print(f"[SIMPLE] GDINO検出: {len(detections)}個 → Centroid追跡: {len(tracks)}個")
                
                # 結果作成（Centroidの結果を使用）
                # 目的: Centroidトラッキング結果からラベル情報を取得する目的で使用
                track_labels = track_result["labels"] if track_result["boxes"] else []
                
                result = {
                    "boxes": tracks[:, :4].tolist() if len(tracks) > 0 else [],
                    "labels": track_labels,
                    "scores": [1.0] * len(tracks),  # スコアは1.0固定
                    "track_ids": tracks[:, 4].tolist() if len(tracks) > 0 else [],  # Track IDは5列目（最後の列）
                    "masks": []
                }
                
                # OpticalFlowは内部でラベル管理を行うため、明示的なクリーンアップは不要
                # 目的: OpticalFlowの簡素な管理機能を活用して効率化する目的で使用
            else:
                # 検出なし時は既存トラックを継続（空の検出でupdateして継続）
                # 目的: GroundingDINO実行間もOpticalFlowトラッキングを継続して滑らかな表示を実現する目的で使用
                print(f"[DEBUG] 検出なし: gdino_result={gdino_result}, should_run_gdino={should_run_gdino}")
                
                # 空の検出でトラッカーを更新（既存トラック継続）
                track_result = self.lang_sam_tracker.track(image_rgb, [], [])
                
                # OpticalFlowが返す結果をそのまま使用
                result = {
                    "boxes": track_result.get("boxes", []),
                    "labels": track_result.get("labels", []),
                    "scores": track_result.get("scores", []),
                    "track_ids": track_result.get("track_ids", []),
                    "masks": track_result.get("masks", [])
                }
                print(f"[SIMPLE] 検出なし → Centroid継続: {len(result['boxes'])}個")
                
            
            # 結果配信
            # 目的: GroundingDINOの純粋な検出結果とCentroidのトラッキング結果を分離して配信する目的で使用
            self._publish_results(gdino_result, result, image_rgb, image_pil, msg.header, bool(gdino_result), should_run_sam2)
            
            if should_run_sam2:
                self.last_sam2_time = current_time
                
        except Exception as e:
            self.logger.error(f"画像処理エラー: {e}")
    
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
                print(f"[DEBUG] GDINO可視化: {len(gdino_only_result['boxes'])}個のボックス（純粋な検出結果）")
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
            centroid_vis = self.lang_sam_tracker.visualize(image_rgb, centroid_only_result)
            print(f"[DEBUG] Centroid可視化: {len(centroid_only_result['boxes'])}個のボックス, Labels: {centroid_only_result['labels']}（トラッキング結果、IDなし）")
            centroid_msg = self.cv_bridge.cv2_to_imgmsg(centroid_vis, encoding='rgb8')
            centroid_msg.header = header
            self.optical_flow_pub.publish(centroid_msg)
            
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
                print(f"[DEBUG] SAM2可視化: {len(sam_only_result['boxes'])}個のボックス, {len([m for m in sam_masks if m is not None])}個のマスク生成")
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