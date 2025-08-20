#!/usr/bin/env python3
"""
LangSAM Tracker Node with Python CSRT Implementation (Fixed Label Mapping)
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

from lang_sam.lang_sam_tracker import LangSAMTracker
from lang_sam.models.utils import DEVICE
from lang_sam.utils import draw_image


class LangSAMTrackerNodeLegacy(Node):
    """Main LangSAM node with Python CSRT tracker integration (Fixed Labels)"""
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node_legacy')
        
        self.get_logger().info("Initializing LangSAMTrackerNode with Python CSRT (Fixed Label Mapping)")
        
        # Initialize CV Bridge
        self.cv_bridge = CvBridge()
        
        # Load parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize LangSAM tracker with Python tracking
        self.tracker = LangSAMTracker(
            sam_type=self.sam_model,
            device=DEVICE
        )
        
        # Setup tracking with Python TrackingManager
        self.tracker.setup_tracking(
            tracking_targets=self.tracking_targets,
            tracking_config={
                'bbox_margin': self.bbox_margin,
                'bbox_min_size': self.bbox_min_size,
                'tracker_min_size': self.tracker_min_size
            },
            csrt_params=self._get_csrt_params()
        )
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscriber
        self._setup_subscriber()
        
        # Frame processing variables
        self.frame_count = 0
        self.last_gdino_time = 0.0
        self.processing_lock = threading.Lock()
        
        self.get_logger().info("LangSAMTrackerNode initialization completed with Python CSRT")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # AI model parameters
        self.declare_parameter('sam_model', 'sam2.1_hiera_tiny')
        self.declare_parameter('text_prompt', 'white line. red pylon. human. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('tracking_targets', ['white line', 'red pylon', 'human', 'car'])
        
        # Execution parameters
        self.declare_parameter('gdino_interval_seconds', 5.0)
        
        # Topic parameters
        self.declare_parameter('input_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('gdino_topic', '/image_gdino')
        self.declare_parameter('csrt_output_topic', '/image_csrt')
        self.declare_parameter('sam_topic', '/image_sam')
        
        # Tracking parameters
        self.declare_parameter('bbox_margin', 5)
        self.declare_parameter('bbox_min_size', 3)
        self.declare_parameter('tracker_min_size', 3)
        
        # CSRT parameters (27 parameters)
        csrt_params = [
            ('csrt_use_hog', True), ('csrt_use_color_names', False),
            ('csrt_use_gray', True), ('csrt_use_rgb', False),
            ('csrt_use_channel_weights', False), ('csrt_use_segmentation', True),
            ('csrt_window_function', 'hann'), ('csrt_kaiser_alpha', 3.75),
            ('csrt_cheb_attenuation', 45.0), ('csrt_template_size', 200.0),
            ('csrt_gsl_sigma', 1.0), ('csrt_hog_orientations', 9),
            ('csrt_hog_clip', 0.2), ('csrt_padding', 3.0),
            ('csrt_filter_lr', 0.02), ('csrt_weights_lr', 0.02),
            ('csrt_num_hog_channels_used', -1), ('csrt_admm_iterations', 4),
            ('csrt_histogram_bins', 16), ('csrt_histogram_lr', 0.04),
            ('csrt_background_ratio', 2), ('csrt_number_of_scales', 33),
            ('csrt_scale_sigma_factor', 0.25), ('csrt_scale_model_max_area', 512.0),
            ('csrt_scale_lr', 0.025), ('csrt_scale_step', 1.02),
            ('csrt_psr_threshold', 0.035)
        ]
        
        for param_name, default_value in csrt_params:
            self.declare_parameter(param_name, default_value)
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        self.sam_model = self.get_parameter('sam_model').value
        self.text_prompt = self.get_parameter('text_prompt').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        self.tracking_targets = self.get_parameter('tracking_targets').value
        
        self.gdino_interval_seconds = self.get_parameter('gdino_interval_seconds').value
        
        self.input_topic = self.get_parameter('input_topic').value
        self.gdino_topic = self.get_parameter('gdino_topic').value
        self.csrt_output_topic = self.get_parameter('csrt_output_topic').value
        self.sam_topic = self.get_parameter('sam_topic').value
        
        self.bbox_margin = self.get_parameter('bbox_margin').value
        self.bbox_min_size = self.get_parameter('bbox_min_size').value
        self.tracker_min_size = self.get_parameter('tracker_min_size').value
        
        self.get_logger().info(f"Loaded parameters: SAM={self.sam_model}, Targets={self.tracking_targets}")
    
    def _get_csrt_params(self):
        """Get CSRT parameters dictionary"""
        csrt_param_names = [
            'csrt_use_hog', 'csrt_use_color_names', 'csrt_use_gray', 'csrt_use_rgb',
            'csrt_use_channel_weights', 'csrt_use_segmentation', 'csrt_window_function',
            'csrt_kaiser_alpha', 'csrt_cheb_attenuation', 'csrt_template_size',
            'csrt_gsl_sigma', 'csrt_hog_orientations', 'csrt_hog_clip', 'csrt_padding',
            'csrt_filter_lr', 'csrt_weights_lr', 'csrt_num_hog_channels_used',
            'csrt_admm_iterations', 'csrt_histogram_bins', 'csrt_histogram_lr',
            'csrt_background_ratio', 'csrt_number_of_scales', 'csrt_scale_sigma_factor',
            'csrt_scale_model_max_area', 'csrt_scale_lr', 'csrt_scale_step', 'csrt_psr_threshold'
        ]
        
        return {param: self.get_parameter(param).value for param in csrt_param_names}
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        self.gdino_publisher = self.create_publisher(Image, self.gdino_topic, 10)
        self.csrt_publisher = self.create_publisher(Image, self.csrt_output_topic, 10)
        self.sam_publisher = self.create_publisher(Image, self.sam_topic, 10)
    
    def _setup_subscriber(self):
        """Setup ROS2 subscriber"""
        self.image_subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )
    
    def image_callback(self, msg: Image):
        """Main image processing callback"""
        with self.processing_lock:
            try:
                # Convert ROS image to OpenCV
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.frame_count += 1
                current_time = time.time()
                
                # Decide processing mode
                run_gdino = (current_time - self.last_gdino_time) >= self.gdino_interval_seconds
                
                if run_gdino:
                    self._process_full_pipeline(cv_image, current_time)
                else:
                    self._process_tracking_only(cv_image)
                    
            except Exception as e:
                self.get_logger().error(f"Image processing error: {e}")
    
    def _process_full_pipeline(self, image: np.ndarray, current_time: float):
        """Process full pipeline: GroundingDINO + CSRT + SAM2"""
        self.get_logger().info(f"Frame {self.frame_count}: Running GroundingDINO detection on image {image.shape}")
        self.last_gdino_time = current_time
        
        try:
            # Convert to PIL for GroundingDINO
            pil_image = self._cv_to_pil(image)
            
            # Run full pipeline with Python TrackingManager
            results = self.tracker.predict_images(
                [pil_image], 
                [self.text_prompt],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                # Publish GroundingDINO results
                self._publish_result(result, image, self.gdino_publisher, "GroundingDINO")
                
                # Publish SAM2 results 
                self._publish_result(result, image, self.sam_publisher, "SAM2")
                
                # CSRT results are handled in tracking-only mode
                self._process_tracking_only(image)
                
        except Exception as e:
            self.get_logger().error(f"Full pipeline processing error: {e}")
    
    def _process_tracking_only(self, image: np.ndarray):
        """Process CSRT tracking only"""
        try:
            # Run Python CSRT tracking (uses fixed TrackingManager)
            result = self.tracker.update_tracking_only(image)
            
            if result and result.get("labels"):
                self.get_logger().debug(f"Frame {self.frame_count}: Python CSRT updated {len(result['labels'])} trackers")
                self._publish_result(result, image, self.csrt_publisher, "CSRT")
                
        except Exception as e:
            self.get_logger().error(f"Tracking-only processing error: {e}")
    
    def _publish_result(self, result: dict, original_image: np.ndarray, publisher, mode: str):
        """Publish visualization result"""
        try:
            boxes = result.get("boxes", np.array([]))
            labels = result.get("labels", [])
            scores = result.get("scores", np.array([]))
            masks = result.get("masks", [])
            
            if len(boxes) == 0:
                return
                
            # Create visualization using fixed draw_image function
            annotated_image = draw_image(
                original_image, masks, boxes, scores, labels
            )
            
            # Convert to ROS message and publish
            ros_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
            ros_msg.header = self.get_clock().now().to_msg()
            publisher.publish(ros_msg)
            
        except Exception as e:
            self.get_logger().error(f"{mode} publish error: {e}")
    
    def _cv_to_pil(self, cv_image: np.ndarray):
        """Convert OpenCV BGR to PIL RGB"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LangSAMTrackerNodeLegacy()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Node error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()