#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import os
import warnings
import contextlib
from PIL import Image as PILImage

# LangSAM統合トラッカー
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lang_sam'))
from lang_sam_tracker import LangSAMTracker
from lang_sam.utils import draw_image


class LangSamWithOptFlowNode(Node):
    """統合パイプライン: GroundingDINO → CSRT → SAM2"""
    
    def __init__(self):
        super().__init__('langsam_with_optflow_node')
        
        # 基本設定
        self.bridge = CvBridge()
        self.frame_count = 0
        
        # config.yamlからパラメータ取得
        self._load_config_parameters()
        
        # パイプライン状態（統合トラッカーで管理）
        self.last_results: Dict = {}
        
        # モデル初期化
        self._init_models()
        
        # ROS通信設定
        self.image_sub = self.create_subscription(
            Image, '/zed/zed_node/rgb/image_rect_color', 
            self.image_callback, 10
        )
        
        # 3つの出力パブリッシャー
        self.gdino_pub = self.create_publisher(Image, '/image_gdino', 10)
        self.csrt_pub = self.create_publisher(Image, '/image_csrt', 10)
        self.sam_pub = self.create_publisher(Image, '/image_sam', 10)
        
        self.get_logger().info("LangSAM分離統合パイプライン開始")
    
    def _load_config_parameters(self):
        """config.yamlからパラメータ読み込み"""
        try:
            # 基本LangSAMパラメータ
            self.declare_parameter('sam_model', 'sam2.1_hiera_small')
            self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
            self.declare_parameter('box_threshold', 0.3)
            self.declare_parameter('text_threshold', 0.2)
            self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
            
            # パイプライン設定
            self.declare_parameter('pipeline_interval', 90)
            
            # CSRTトラッカー設定
            self.declare_parameter('min_bbox_size', 20)
            self.declare_parameter('max_bbox_ratio', 0.8)
            
            # CUDA設定
            self.declare_parameter('cuda_arch_list', '6.0;6.1;7.0;7.5;8.0;8.6')
            self.declare_parameter('enable_torch_no_grad', True)
            self.declare_parameter('enable_gpu_memory_cleanup', True)
            
            # ログ設定
            self.declare_parameter('enable_fps_logging', True)
            self.declare_parameter('enable_debug_logging', False)
            
            # パラメータ取得
            self.sam_model = self.get_parameter('sam_model').value
            self.text_prompt = self.get_parameter('text_prompt').value
            self.box_threshold = self.get_parameter('box_threshold').value
            self.text_threshold = self.get_parameter('text_threshold').value
            self.tracking_targets = self.get_parameter('tracking_targets').value
            self.pipeline_interval = self.get_parameter('pipeline_interval').value
            self.min_bbox_size = self.get_parameter('min_bbox_size').value
            self.max_bbox_ratio = self.get_parameter('max_bbox_ratio').value
            self.cuda_arch_list = self.get_parameter('cuda_arch_list').value
            self.enable_torch_no_grad = self.get_parameter('enable_torch_no_grad').value
            self.enable_gpu_memory_cleanup = self.get_parameter('enable_gpu_memory_cleanup').value
            self.enable_fps_logging = self.get_parameter('enable_fps_logging').value
            self.enable_debug_logging = self.get_parameter('enable_debug_logging').value
            
            self.get_logger().info(f"設定読み込み完了: {self.sam_model}, prompt: {self.text_prompt}")
            
        except Exception as e:
            self.get_logger().error(f"config.yaml読み込みエラー: {e}")
            raise RuntimeError(f"設定読み込み失敗: {e}")
    
    def _init_models(self):
        """LangSAMトラッカー初期化"""
        try:
            # CUDA環境設定
            self._configure_cuda_environment()
            
            # LangSAM統合トラッカー初期化
            self.get_logger().info(f"LangSAMトラッカー初期化中... ({self.sam_model})")
            self.tracker_model = LangSAMTracker(sam_type=self.sam_model)
            
            # トラッキング対象設定
            self.tracker_model.set_tracking_targets(self.tracking_targets)
            
            self.get_logger().info("LangSAMトラッカー初期化完了")
            
        except Exception as e:
            self.get_logger().error(f"LangSAMトラッカー初期化失敗: {e}")
            raise RuntimeError(f"LangSAMトラッカー初期化失敗: {e}")
    
    def _configure_cuda_environment(self):
        """CUDA環境設定"""
        try:
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
            if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
                os.environ['TORCH_CUDA_ARCH_LIST'] = self.cuda_arch_list
            warnings.filterwarnings("ignore", category=UserWarning)
            
            if torch.cuda.is_available():
                self.get_logger().info(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
                if self.enable_gpu_memory_cleanup:
                    torch.cuda.empty_cache()
            else:
                self.get_logger().warn("CUDA利用不可")
                
        except Exception as e:
            self.get_logger().warn(f"CUDA設定エラー: {e}")
    
    def image_callback(self, msg: Image):
        """メイン処理コールバック"""
        try:
            # 画像変換
            image_cv = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
            
            # パイプライン実行判定
            if self.frame_count % self.pipeline_interval == 0:
                self._run_integrated_pipeline(image_cv)
            else:
                # トラッカーのみ更新
                self._update_trackers_only(image_cv)
            
        except Exception as e:
            self.get_logger().error(f"処理エラー: {e}")
    
    def _run_integrated_pipeline(self, image: np.ndarray):
        """統合パイプライン: GroundingDINO → CSRT → SAM2"""
        try:
            self.get_logger().info("統合パイプライン開始")
            
            # BGR → RGB変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # 統合予測実行（GDINO→CSRT→SAM2）
            context_manager = torch.no_grad() if self.enable_torch_no_grad else contextlib.nullcontext()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with context_manager:
                    results = self.tracker_model.predict_with_tracking(
                        images_pil=[pil_image],
                        texts_prompt=[self.text_prompt],
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        update_trackers=True,
                        run_sam=True
                    )
            
            if results and len(results) > 0:
                result = results[0]
                self.last_results = result
                
                # 3つの出力を順次配信
                self._publish_all_results(image, result)
                
                boxes_count = len(result.get('boxes', []))
                masks_count = len(result.get('masks', []))
                self.get_logger().info(f"統合パイプライン完了: {boxes_count}個検出, {masks_count}個マスク")
            else:
                self.get_logger().warn("統合パイプライン: 結果なし")
            
        except Exception as e:
            self.get_logger().error(f"統合パイプライン実行エラー: {e}")
    
    def _update_trackers_only(self, image: np.ndarray):
        """トラッカーのみ更新（検出なし）"""
        try:
            # トラッカーのみ更新
            result = self.tracker_model.update_trackers_only(image)
            
            if result and len(result.get('boxes', [])) > 0:
                self.last_results = result
                self._publish_csrt_result(image, result)
                
        except Exception as e:
            self.get_logger().error(f"トラッカー更新エラー: {e}")
    
    def _publish_all_results(self, image: np.ndarray, result: Dict):
        """3つの出力を順次配信"""
        try:
            boxes = result.get('boxes', [])
            labels = result.get('labels', [])
            scores = result.get('scores', [])
            masks = result.get('masks', [])
            mask_scores = result.get('mask_scores', [])
            
            # GroundingDINO結果配信
            self._publish_gdino_result(image, boxes, labels, scores)
            
            # CSRT結果配信
            self._publish_csrt_result(image, result)
            
            # SAM2結果配信
            if len(masks) > 0:
                self._publish_sam2_result(image, masks, mask_scores, boxes, labels)
            
        except Exception as e:
            self.get_logger().error(f"結果配信エラー: {e}")
    
    
    
    def _publish_gdino_result(self, image: np.ndarray, boxes: List, labels: List, scores: List):
        """GroundingDINO結果配信（/image_gdino）"""
        try:
            if len(boxes) == 0:
                # 空の結果の場合は元画像を配信
                gdino_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                self.gdino_pub.publish(gdino_msg)
                return
                
            height, width = image.shape[:2]
            dummy_masks = np.zeros((len(boxes), height, width), dtype=bool)
            
            # BGR → RGB変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            gdino_image = draw_image(
                image_rgb=image_rgb,
                masks=dummy_masks,
                xyxy=np.array(boxes),
                probs=np.array(scores),
                labels=labels
            )
            
            # RGB → BGR変換
            gdino_image_bgr = cv2.cvtColor(gdino_image, cv2.COLOR_RGB2BGR)
            gdino_msg = self.bridge.cv2_to_imgmsg(gdino_image_bgr, 'bgr8')
            self.gdino_pub.publish(gdino_msg)
            
        except Exception as e:
            self.get_logger().error(f"GroundingDINO配信エラー: {e}")
    
    def _publish_csrt_result(self, image: np.ndarray, result: Dict):
        """CSRT結果配信（/image_csrt）"""
        try:
            csrt_image = image.copy()
            
            boxes = result.get('boxes', [])
            labels = result.get('labels', [])
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if len(box) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(csrt_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(csrt_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            csrt_msg = self.bridge.cv2_to_imgmsg(csrt_image, 'bgr8')
            self.csrt_pub.publish(csrt_msg)
            
        except Exception as e:
            self.get_logger().error(f"CSRT配信エラー: {e}")
    
    def _publish_sam2_result(self, image: np.ndarray, masks: List, mask_scores: List, boxes: List, labels: List):
        """SAM2結果配信（/image_sam）"""
        try:
            if len(masks) == 0 or len(boxes) == 0:
                # 空の結果の場合は元画像を配信
                sam_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                self.sam_pub.publish(sam_msg)
                return
                
            # BGR → RGB変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            sam_image = draw_image(
                image_rgb=image_rgb,
                masks=np.array(masks),
                xyxy=np.array(boxes),
                probs=np.array(mask_scores),
                labels=labels
            )
            
            # RGB → BGR変換
            sam_image_bgr = cv2.cvtColor(sam_image, cv2.COLOR_RGB2BGR)
            sam_msg = self.bridge.cv2_to_imgmsg(sam_image_bgr, 'bgr8')
            self.sam_pub.publish(sam_msg)
            
        except Exception as e:
            self.get_logger().error(f"SAM2配信エラー: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LangSamWithOptFlowNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()