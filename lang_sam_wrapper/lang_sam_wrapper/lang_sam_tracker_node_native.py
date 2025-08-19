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

from lang_sam.lang_sam_tracker import LangSAMTracker
from lang_sam.models.utils import DEVICE
from lang_sam.tracker_utils.csrt_native_client import CSRTNativeClient
from lang_sam.utils import draw_image


class LangSAMTrackerNodeNative(Node):
    """Main LangSAM node with native C++ CSRT tracker integration"""
    
    def __init__(self):
        super().__init__('lang_sam_tracker_node_native')
        
        self.get_logger().info("Initializing LangSAMTrackerNode with Native C++ CSRT")
        
        # Initialize CV Bridge
        self.cv_bridge = CvBridge()
        
        # Load parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize native C++ CSRT client
        self.native_client = CSRTNativeClient(self)
        
        if not self.native_client.is_available():
            self.get_logger().error("Native C++ CSRT extension not available, shutting down")
            return
        
        # Initialize LangSAM tracker (detection and segmentation only)
        self.tracker = LangSAMTracker(
            sam_type=self.sam_model,
            device=DEVICE
        )
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        
        # Timing variables
        self.last_gdino_time = 0.0
        self.frame_count = 0
        
        # Store detection labels for CSRT/SAM visualization
        self.current_detection_labels = []
        
        # Thread safety
        self.processing_lock = threading.Lock()
        
        self.get_logger().info("LangSAM Tracker Node with Native C++ CSRT initialized successfully")
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
        
        # All CSRT parameters are already declared by CSRTNativeClient
    
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
        """Log current configuration"""
        self.get_logger().info("=== LangSAM Native Configuration ===")
        self.get_logger().info(f"SAM Model: {self.sam_model}")
        self.get_logger().info(f"Text Prompt: {self.text_prompt}")
        self.get_logger().info(f"Box Threshold: {self.box_threshold}")
        self.get_logger().info(f"Text Threshold: {self.text_threshold}")
        self.get_logger().info(f"GDINO Interval: {self.gdino_interval_seconds}s")
        self.get_logger().info(f"Input Topic: {self.input_topic}")
        self.get_logger().info(f"Tracking Targets: {self.tracking_targets}")
        self.get_logger().info(f"Native C++ CSRT: {self.native_client.is_available()}")
        self.get_logger().info(f"Native Trackers: {self.native_client.get_tracker_count()}")
        self.get_logger().info("=== End Configuration ===")
    
    def image_callback(self, msg: Image):
        """Process incoming image"""
        with self.processing_lock:
            try:
                # Convert ROS image to OpenCV
                image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                current_time = time.time()
                self.frame_count += 1
                
                # Determine if GroundingDINO should run
                should_run_gdino = (current_time - self.last_gdino_time) >= self.gdino_interval_seconds
                
                if should_run_gdino:
                    self.get_logger().info(f"Frame {self.frame_count}: Running GroundingDINO detection on image {image.shape}")
                    
                    # Convert image to PIL format for LangSAMTracker
                    from PIL import Image as PILImage
                    try:
                        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        self.get_logger().debug(f"PIL image size: {pil_image.size}")
                    except Exception as e:
                        self.get_logger().error(f"Image conversion failed: {e}")
                        return
                    
                    # Run full prediction pipeline (GroundingDINO + built-in tracking + SAM)
                    results = self.tracker.predict_with_tracking(
                        images_pil=[pil_image],
                        texts_prompt=[self.text_prompt],
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        update_trackers=False,  # Disable built-in tracking
                        run_sam=True  # Enable SAM for segmentation
                    )
                    
                    # Extract detections from results - Fixed format parsing
                    detections = []
                    self.get_logger().info(f"GroundingDINO results length: {len(results) if results else 0}")
                    
                    if results and len(results) > 0:
                        result = results[0]  # First image result
                        self.get_logger().info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                        
                        # GroundingDINO returns 'boxes', 'labels', 'scores' keys
                        boxes = result.get("boxes", [])
                        labels = result.get("labels", [])
                        scores = result.get("scores", [])
                        
                        if len(boxes) > 0 and len(labels) > 0:
                            self.get_logger().info(f"Found {len(boxes)} detections")
                            for i, (box, label) in enumerate(zip(boxes, labels)):
                                class Detection:
                                    def __init__(self, box, label, score=1.0):
                                        # box format: [x1, y1, x2, y2] -> convert to [x, y, w, h]
                                        self.x = int(box[0])
                                        self.y = int(box[1]) 
                                        self.width = int(box[2] - box[0])
                                        self.height = int(box[3] - box[1])
                                        self.label = label
                                        self.score = score
                                
                                score = scores[i] if i < len(scores) else 1.0
                                detection = Detection(box, label, score)
                                detections.append(detection)
                                self.get_logger().info(f"Detection {i}: {label} at ({detection.x}, {detection.y}, {detection.width}, {detection.height})")
                        else:
                            self.get_logger().warn(f"No detections found: boxes={len(boxes)}, labels={len(labels)}")
                    else:
                        self.get_logger().warn("Empty or no results from GroundingDINO")
                    
                    if detections:
                        self.get_logger().info(f"GroundingDINO detected {len(detections)} objects")
                        
                        # Initialize native C++ CSRT trackers
                        detection_boxes = []
                        detection_labels = []
                        
                        for det in detections:
                            if hasattr(det, 'label') and det.label in self.tracking_targets:
                                bbox = (det.x, det.y, det.width, det.height)
                                detection_boxes.append(bbox)
                                detection_labels.append(det.label)
                        
                        # Process detections with native C++ CSRT
                        if detection_boxes:
                            # Store detection labels for CSRT/SAM visualization
                            self.current_detection_labels = detection_labels.copy()
                            
                            native_results = self.native_client.process_detections(
                                image, detection_boxes, detection_labels
                            )
                            self.get_logger().info(f"Native C++ CSRT initialized {len(native_results)} trackers")
                    
                    # Create GroundingDINO visualization using draw_image
                    if detections:
                        # Convert detections to format for draw_image
                        xyxy = np.array([[det.x, det.y, det.x + det.width, det.y + det.height] for det in detections])
                        labels = [det.label for det in detections]
                        probs = np.array([getattr(det, 'score', 1.0) for det in detections])
                        masks = np.zeros((len(detections), image.shape[0], image.shape[1]), dtype=bool)  # Empty masks
                        
                        # Convert BGR to RGB for draw_image
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        gdino_image_rgb = draw_image(image_rgb, masks, xyxy, probs, labels)
                        gdino_image = cv2.cvtColor(gdino_image_rgb, cv2.COLOR_RGB2BGR)
                    else:
                        gdino_image = image.copy()
                        cv2.putText(gdino_image, "GroundingDINO: No detections", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    gdino_msg = self.cv_bridge.cv2_to_imgmsg(gdino_image, 'bgr8')
                    gdino_msg.header = msg.header
                    self.gdino_pub.publish(gdino_msg)
                    
                    self.last_gdino_time = current_time
                
                else:
                    # Only run native C++ CSRT tracking
                    native_results = self.native_client.update_trackers(image)
                    
                    if native_results:
                        self.get_logger().debug(f"Frame {self.frame_count}: Native C++ CSRT updated {len(native_results)} trackers")
                        
                        # Get labels from native C++ tracker (fixed order)
                        tracker_labels = self.native_client.get_tracker_labels()
                        self.get_logger().debug(f"[Native Node] C++ tracker returned {len(tracker_labels)} labels: {tracker_labels}")
                        
                        # Convert native results to detection format for SAM
                        csrt_detections = []
                        for i, bbox_tuple in enumerate(native_results):
                            x, y, w, h = bbox_tuple
                            # Use labels from C++ tracker (maintains correct order)
                            label = tracker_labels[i] if i < len(tracker_labels) else "tracked"
                            
                            # Create detection-like object for compatibility
                            class CSRTDetection:
                                def __init__(self, x, y, w, h, label):
                                    self.x, self.y, self.width, self.height = x, y, w, h
                                    self.label = label
                            
                            csrt_detections.append(CSRTDetection(x, y, w, h, label))
                            self.get_logger().debug(f"[Native Node] CSRT Detection {i}: {label} at ({x}, {y}, {w}, {h})")
                        
                        # Create CSRT visualization using draw_image
                        xyxy = np.array([[det.x, det.y, det.x + det.width, det.y + det.height] for det in csrt_detections])
                        labels = [det.label for det in csrt_detections]
                        probs = np.ones(len(csrt_detections))  # All tracking results have confidence 1.0
                        masks = np.zeros((len(csrt_detections), image.shape[0], image.shape[1]), dtype=bool)  # Empty masks
                        
                        # Convert BGR to RGB for draw_image
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        csrt_image_rgb = draw_image(image_rgb, masks, xyxy, probs, labels)
                        csrt_image = cv2.cvtColor(csrt_image_rgb, cv2.COLOR_RGB2BGR)
                        csrt_msg = self.cv_bridge.cv2_to_imgmsg(csrt_image, 'bgr8')
                        csrt_msg.header = msg.header
                        self.csrt_pub.publish(csrt_msg)
                        
                        # Run SAM2 segmentation directly on CSRT bboxes
                        try:
                            # Convert to RGB format for SAM
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            # Convert CSRT detection boxes to xyxy format for SAM
                            sam_boxes = []
                            for det in csrt_detections:
                                # Convert (x, y, w, h) to (x1, y1, x2, y2)
                                x1, y1 = det.x, det.y
                                x2, y2 = det.x + det.width, det.y + det.height
                                sam_boxes.append([x1, y1, x2, y2])
                            
                            if sam_boxes:
                                sam_boxes_array = np.array(sam_boxes)
                                
                                # Direct SAM prediction on CSRT bboxes (no GroundingDINO)
                                sam_model = self.tracker.coordinator.sam
                                masks, scores, logits = sam_model.predict(image_rgb, sam_boxes_array)
                                
                                # Use CSRT labels and SAM masks for visualization
                                labels = [det.label for det in csrt_detections]
                                xyxy = sam_boxes_array
                                
                                # Fix scores dimension: flatten to 1D if needed
                                if scores is not None:
                                    scores = np.array(scores)
                                    if scores.ndim > 1:
                                        scores = scores.flatten()
                                    probs = scores
                                else:
                                    probs = np.ones(len(csrt_detections))
                                
                                # Convert masks to proper format for draw_image
                                if masks is not None and len(masks) > 0:
                                    if len(masks.shape) == 3:
                                        # masks shape: (N, H, W) -> convert to bool
                                        masks = masks.astype(bool)
                                    sam_image_rgb = draw_image(image_rgb, masks, xyxy, probs, labels)
                                else:
                                    # No masks, use bounding boxes only
                                    empty_masks = np.zeros((len(csrt_detections), image.shape[0], image.shape[1]), dtype=bool)
                                    sam_image_rgb = draw_image(image_rgb, empty_masks, xyxy, probs, labels)
                                
                                sam_image = cv2.cvtColor(sam_image_rgb, cv2.COLOR_RGB2BGR)
                            else:
                                # No CSRT detections, show empty image
                                sam_image = image.copy()
                                cv2.putText(sam_image, "SAM: No CSRT detections", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            
                            sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_image, 'bgr8')
                            sam_msg.header = msg.header
                            self.sam_pub.publish(sam_msg)
                            
                        except Exception as e:
                            self.get_logger().error(f"SAM2 segmentation failed: {e}")
                            # Fallback visualization with bounding boxes
                            xyxy = np.array([[det.x, det.y, det.x + det.width, det.y + det.height] for det in csrt_detections])
                            labels = [det.label for det in csrt_detections]
                            probs = np.ones(len(csrt_detections))
                            masks = np.zeros((len(csrt_detections), image.shape[0], image.shape[1]), dtype=bool)
                            
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            sam_image_rgb = draw_image(image_rgb, masks, xyxy, probs, labels)
                            sam_image = cv2.cvtColor(sam_image_rgb, cv2.COLOR_RGB2BGR)
                            sam_msg = self.cv_bridge.cv2_to_imgmsg(sam_image, 'bgr8')
                            sam_msg.header = msg.header
                            self.sam_pub.publish(sam_msg)
                    
                    else:
                        # No trackers active, publish empty images with text overlay
                        empty_image = image.copy()
                        cv2.putText(empty_image, "No Active Trackers", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        empty_msg = self.cv_bridge.cv2_to_imgmsg(empty_image, 'bgr8')
                        empty_msg.header = msg.header
                        self.csrt_pub.publish(empty_msg)
                        self.sam_pub.publish(empty_msg)
                
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("Shutting down LangSAM Native Node")
        if hasattr(self, 'native_client'):
            self.native_client.clear_trackers()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LangSAMTrackerNodeNative()
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