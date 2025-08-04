#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import os
import warnings
from PIL import Image as PILImage

# LangSAM統合トラッカー
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lang_sam'))
from lang_sam_tracker import LangSAMTracker
from lang_sam.utils import draw_image


class LangSamWithOptFlowNode(Node):
    """シンプル統合トラッカー: GroundingDINO → CSRT → SAM2"""
    
    def __init__(self):
        super().__init__('langsam_with_optflow_node')
        
        # 基本設定
        self.bridge = CvBridge()
        self.frame_count = 0
        
        # 全パラメータをconfig.yamlから読み込み
        self._declare_all_parameters()
        self._load_all_parameters()
        
        # CUDA環境設定
        self._setup_cuda()
        
        # 統合トラッカー初期化
        self.tracker = LangSAMTracker(sam_type=self.sam_model)
        self.tracker.set_tracking_targets(self.tracking_targets)
        
        # ROS通信設定（パラメータ化）
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )
        
        # 3つの出力パブリッシャー（パラメータ化）
        self.gdino_pub = self.create_publisher(Image, self.gdino_topic, 10)
        self.csrt_pub = self.create_publisher(Image, self.csrt_topic, 10)
        self.sam_pub = self.create_publisher(Image, self.sam_topic, 10)
        
        self.get_logger().info(f"シンプル統合トラッカー開始: {self.sam_model}")
    
    def _declare_all_parameters(self):
        """全パラメータ宣言"""
        # モデル設定
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.2)
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # 実行設定
        self.declare_parameter('pipeline_interval', 90)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('enable_sam', True)
        
        # トピック設定
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # CUDA設定
        self.declare_parameter('cuda_arch_list', '6.0;6.1;7.0;7.5;8.0;8.6')
        self.declare_parameter('enable_torch_no_grad', True)
        
        # ログ設定
        self.declare_parameter('enable_debug', False)
    
    def _load_all_parameters(self):
        """全パラメータ読み込み"""
        # モデル設定
        self.sam_model = self.get_parameter('sam_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        self.tracking_targets = self.get_parameter('tracking_targets').value
        
        # 実行設定
        self.pipeline_interval = self.get_parameter('pipeline_interval').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.enable_sam = self.get_parameter('enable_sam').value
        
        # トピック設定
        self.input_topic = self.get_parameter('input_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.csrt_topic = self.get_parameter('csrt_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
        
        # CUDA設定
        self.cuda_arch_list = self.get_parameter('cuda_arch_list').value
        self.enable_torch_no_grad = self.get_parameter('enable_torch_no_grad').value
        
        # ログ設定
        self.enable_debug = self.get_parameter('enable_debug').value
    
    def _setup_cuda(self):
        """CUDA環境設定"""
        os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
        if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
            os.environ['TORCH_CUDA_ARCH_LIST'] = self.cuda_arch_list
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.enable_debug:
                self.get_logger().info(f"CUDA: {torch.cuda.get_device_name(0)}")
    
    def image_callback(self, msg: Image):
        """メイン処理"""
        try:
            # 画像変換
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
            
            # パイプライン実行判定
            if self.frame_count % self.pipeline_interval == 0:
                self._run_full_pipeline(image)
            elif self.enable_tracking:
                self._run_tracking_only(image)
                
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f"処理エラー: {e}")
    
    def _run_full_pipeline(self, image: np.ndarray):
        """完全パイプライン実行"""
        try:
            # BGR → RGB → PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # 統合予測実行
            context = torch.no_grad() if self.enable_torch_no_grad else torch.enable_grad()
            with warnings.catch_warnings(), context:
                warnings.simplefilter("ignore")
                results = self.tracker.predict_with_tracking(
                    images_pil=[pil_image],
                    texts_prompt=[self.text_prompt],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    update_trackers=self.enable_tracking,
                    run_sam=self.enable_sam
                )
            
            # 結果配信
            if results and len(results) > 0:
                result = results[0]
                self._publish_all_results(image, result)
                
                if self.enable_debug:
                    boxes_count = len(result.get('boxes', []))
                    masks_count = len(result.get('masks', []))
                    self.get_logger().info(f"完全パイプライン完了: {boxes_count}個検出, {masks_count}個マスク")
                    
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f"完全パイプライン失敗: {e}")
    
    def _run_tracking_only(self, image: np.ndarray):
        """トラッキングのみ実行"""
        try:
            result = self.tracker.update_trackers_only(image)
            if result and len(result.get('boxes', [])) > 0:
                self._publish_tracking(image, result.get('boxes', []), result.get('labels', []), self.csrt_pub)
                
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f"トラッキング失敗: {e}")
    
    def _publish_all_results(self, image: np.ndarray, result: dict):
        """全結果配信"""
        boxes = result.get('boxes', [])
        labels = result.get('labels', [])
        scores = result.get('scores', [])
        masks = result.get('masks', [])
        mask_scores = result.get('mask_scores', [])
        
        # GroundingDINO結果
        self._publish_detection(image, boxes, labels, scores, self.gdino_pub)
        
        # CSRT結果
        self._publish_tracking(image, boxes, labels, self.csrt_pub)
        
        # SAM2結果
        if len(masks) > 0:
            self._publish_segmentation(image, masks, mask_scores, boxes, labels, self.sam_pub)
    
    def _publish_detection(self, image: np.ndarray, boxes, labels, scores, publisher):
        """検出結果配信"""
        try:
            if len(boxes) == 0:
                msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                publisher.publish(msg)
                return
            
            height, width = image.shape[:2]
            dummy_masks = np.zeros((len(boxes), height, width), dtype=bool)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result_image = draw_image(
                image_rgb=image_rgb,
                masks=dummy_masks,
                xyxy=np.array(boxes),
                probs=np.array(scores),
                labels=labels
            )
            
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8')
            publisher.publish(msg)
            
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f"検出配信失敗: {e}")
    
    def _publish_tracking(self, image: np.ndarray, boxes, labels, publisher):
        """トラッキング結果配信"""
        try:
            result_image = image.copy()
            
            for box, label in zip(boxes, labels):
                if len(box) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_image, str(label), (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            msg = self.bridge.cv2_to_imgmsg(result_image, 'bgr8')
            publisher.publish(msg)
            
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f"トラッキング配信失敗: {e}")
    
    def _publish_segmentation(self, image: np.ndarray, masks, mask_scores, boxes, labels, publisher):
        """セグメンテーション結果配信"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result_image = draw_image(
                image_rgb=image_rgb,
                masks=np.array(masks),
                xyxy=np.array(boxes),
                probs=np.array(mask_scores),
                labels=labels
            )
            
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(result_bgr, 'bgr8')
            publisher.publish(msg)
            
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f"セグメンテーション配信失敗: {e}")


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