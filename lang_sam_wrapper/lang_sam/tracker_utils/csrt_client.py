#!/usr/bin/env python3
"""
Native C++ CSRT Tracker Client with ROS Parameter Integration
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import numpy as np
import cv2

try:
    # Import C++ extension
    from lang_sam_wrapper.csrt_native import CSRTParams, CSRTTrackerNative, CSRTManagerNative
    NATIVE_AVAILABLE = True
    print("[CSRTClient] C++ CSRT extension loaded successfully")
except ImportError as e:
    print(f"[CSRTClient] C++ extension not available: {e}")
    NATIVE_AVAILABLE = False

class CSRTClient:
    """Client for C++ CSRT tracker with ROS parameter integration"""
    
    def __init__(self, node: Node):
        self.node = node
        self.logger = node.get_logger()
        self._cached_labels = []  # Cache for labels to avoid redundant calls
        
        if not NATIVE_AVAILABLE:
            self.logger.error("C++ CSRT extension not available")
            self.manager = None
            return
            
        # Initialize with ROS parameters
        self.params = self._load_params_from_ros()
        self.manager = CSRTManagerNative(self.params)
        
        # Log parameter loading
        self._log_parameters()
        
        self.logger.info("CSRTNativeClient initialized with native C++ CSRT")
    
    def _load_params_from_ros(self) -> 'CSRTParams':
        """Load CSRT parameters from ROS parameters"""
        params = CSRTParams()
        
        # Declare and get parameters with defaults
        param_mapping = {
            'csrt_use_hog': (params.use_hog, bool),
            'csrt_use_color_names': (params.use_color_names, bool),
            'csrt_use_gray': (params.use_gray, bool),
            'csrt_use_rgb': (params.use_rgb, bool),
            'csrt_use_channel_weights': (params.use_channel_weights, bool),
            'csrt_use_segmentation': (params.use_segmentation, bool),
            'csrt_window_function': (params.window_function, str),
            'csrt_kaiser_alpha': (params.kaiser_alpha, float),
            'csrt_cheb_attenuation': (params.cheb_attenuation, float),
            'csrt_template_size': (params.template_size, float),
            'csrt_gsl_sigma': (params.gsl_sigma, float),
            'csrt_hog_orientations': (params.hog_orientations, int),
            'csrt_hog_clip': (params.hog_clip, float),
            'csrt_padding': (params.padding, float),
            'csrt_filter_lr': (params.filter_lr, float),
            'csrt_weights_lr': (params.weights_lr, float),
            'csrt_num_hog_channels_used': (params.num_hog_channels_used, int),
            'csrt_admm_iterations': (params.admm_iterations, int),
            'csrt_histogram_bins': (params.histogram_bins, int),
            'csrt_histogram_lr': (params.histogram_lr, float),
            'csrt_background_ratio': (params.background_ratio, int),
            'csrt_number_of_scales': (params.number_of_scales, int),
            'csrt_scale_sigma_factor': (params.scale_sigma_factor, float),
            'csrt_scale_model_max_area': (params.scale_model_max_area, float),
            'csrt_scale_lr': (params.scale_lr, float),
            'csrt_scale_step': (params.scale_step, float),
            'csrt_psr_threshold': (params.psr_threshold, float),
            # CSRT復旧機能パラメータ
            'enable_csrt_recovery': (params.enable_recovery, bool),
            'frame_buffer_duration': (params.buffer_duration, float),
            'time_travel_seconds': (params.time_travel_seconds, float),
            'fast_forward_frames': (params.fast_forward_frames, int),
        }
        
        for param_name, (default_value, param_type) in param_mapping.items():
            try:
                # Declare parameter
                self.node.declare_parameter(param_name, default_value)
                # Get parameter value
                value = self.node.get_parameter(param_name).value
                
                # Set to CSRTParams object
                attr_name = param_name.replace('csrt_', '')
                setattr(params, attr_name, value)
                
                self.logger.debug(f"Loaded parameter {param_name}: {value}")
                
            except Exception as e:
                self.logger.warn(f"Failed to load parameter {param_name}: {e}, using default: {default_value}")
                
        return params
    
    def _log_parameters(self):
        """Log all CSRT parameters for verification"""
        self.logger.info("=== Native C++ CSRT Parameters ===")
        self.logger.info(f"use_hog: {self.params.use_hog}")
        self.logger.info(f"use_color_names: {self.params.use_color_names}")
        self.logger.info(f"use_gray: {self.params.use_gray}")
        self.logger.info(f"use_rgb: {self.params.use_rgb}")
        self.logger.info(f"use_channel_weights: {self.params.use_channel_weights}")
        self.logger.info(f"use_segmentation: {self.params.use_segmentation}")
        self.logger.info(f"window_function: {self.params.window_function}")
        self.logger.info(f"kaiser_alpha: {self.params.kaiser_alpha}")
        self.logger.info(f"cheb_attenuation: {self.params.cheb_attenuation}")
        self.logger.info(f"template_size: {self.params.template_size}")
        self.logger.info(f"gsl_sigma: {self.params.gsl_sigma}")
        self.logger.info(f"hog_orientations: {self.params.hog_orientations}")
        self.logger.info(f"hog_clip: {self.params.hog_clip}")
        self.logger.info(f"padding: {self.params.padding}")
        self.logger.info(f"filter_lr: {self.params.filter_lr}")
        self.logger.info(f"weights_lr: {self.params.weights_lr}")
        self.logger.info(f"num_hog_channels_used: {self.params.num_hog_channels_used}")
        self.logger.info(f"admm_iterations: {self.params.admm_iterations}")
        self.logger.info(f"histogram_bins: {self.params.histogram_bins}")
        self.logger.info(f"histogram_lr: {self.params.histogram_lr}")
        self.logger.info(f"background_ratio: {self.params.background_ratio}")
        self.logger.info(f"number_of_scales: {self.params.number_of_scales}")
        self.logger.info(f"scale_sigma_factor: {self.params.scale_sigma_factor}")
        self.logger.info(f"scale_model_max_area: {self.params.scale_model_max_area}")
        self.logger.info(f"scale_lr: {self.params.scale_lr}")
        self.logger.info(f"scale_step: {self.params.scale_step}")
        self.logger.info(f"psr_threshold: {self.params.psr_threshold}")
        self.logger.info("=== CSRT復旧機能パラメータ ===")
        self.logger.info(f"enable_recovery: {self.params.enable_recovery}")
        self.logger.info(f"buffer_duration: {self.params.buffer_duration}")
        self.logger.info(f"time_travel_seconds: {self.params.time_travel_seconds}")
        self.logger.info(f"fast_forward_frames: {self.params.fast_forward_frames}")
        self.logger.info("=== End CSRT Parameters ===")
    
    def update_parameters_from_ros(self):
        """Update parameters from current ROS parameter values"""
        if not self.manager:
            return
            
        self.params = self._load_params_from_ros()
        self.manager.set_default_params(self.params)
        self.logger.info("Updated CSRT parameters from ROS parameter server")
    
    def process_detections(self, image: np.ndarray, detections, labels):
        """Process detections using native C++ CSRT"""
        if not self.manager:
            self.logger.error("Native C++ manager not available")
            return []
            
        try:
            self.logger.debug(f"[CSRTNativeClient] Processing {len(detections)} detections with labels: {labels}")
            
            # Convert detections to proper format
            detection_rects = []
            for det in detections:
                if hasattr(det, 'x'):  # xyxy format
                    rect = (det.x, det.y, det.width, det.height)
                else:  # tuple format
                    rect = det
                detection_rects.append(rect)
            
            results = self.manager.process_detections(image, detection_rects, labels)
            self.logger.info(f"[CSRTNativeClient] Native C++ CSRT processed {len(results)} detections, created {len(results)} trackers")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in native C++ detection processing: {e}")
            return []
    
    def update_trackers(self, image: np.ndarray):
        """Update all trackers using C++ CSRT and cache labels"""
        if not self.manager:
            return []
            
        try:
            results = self.manager.update_trackers(image)
            # Cache labels immediately after update to avoid redundant calls
            self._cached_labels = self.manager.get_tracker_labels() if results else []
            self.logger.debug(f"[CSRTClient] C++ CSRT updated {len(results)} trackers with labels: {self._cached_labels}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in C++ tracker update: {e}")
            return []
    
    def update_trackers_with_recovery(self, image: np.ndarray):
        """Update all trackers using C++ CSRT with recovery functionality"""
        if not self.manager:
            return []
            
        try:
            # 復旧機能付き更新を使用
            if hasattr(self.manager, 'update_trackers_with_recovery'):
                results = self.manager.update_trackers_with_recovery(image)
                # Cache labels immediately after update
                self._cached_labels = self.manager.get_tracker_labels() if results else []
                self.logger.debug(f"[CSRTClient] C++ CSRT復旧機能更新 {len(results)} trackers with labels: {self._cached_labels}")
                return results
            else:
                # フォールバック: 通常の更新
                return self.update_trackers(image)
            
        except Exception as e:
            self.logger.error(f"Error in C++ tracker recovery update: {e}")
            return []
    
    def clear_trackers(self):
        """Clear all trackers"""
        if self.manager:
            self.manager.clear_trackers()
            self.logger.info("Cleared all native C++ CSRT trackers")
    
    def get_tracker_count(self) -> int:
        """Get current tracker count"""
        if self.manager:
            return self.manager.get_tracker_count()
        return 0
    
    def get_cached_labels(self) -> list:
        """Get cached tracker labels (avoids redundant C++ calls)"""
        return self._cached_labels
    
    def get_tracker_labels(self) -> list:
        """Get current tracker labels from C++ (direct call)"""
        if self.manager:
            labels = self.manager.get_tracker_labels()
            self.logger.debug(f"[CSRTClient] Retrieved {len(labels)} labels from C++: {labels}")
            return labels
        return []
    
    def get_recovery_stats(self) -> dict:
        """Get recovery statistics from C++ manager"""
        if not self.manager:
            return {'recovered': 0, 'failed': 0, 'recovery_enabled': False}
            
        try:
            return {
                'recovered': getattr(self.manager, 'get_recovered_tracker_count', lambda: 0)(),
                'failed': getattr(self.manager, 'get_failed_tracker_count', lambda: 0)(),
                'recovery_enabled': getattr(self.manager, 'is_recovery_enabled', lambda: False)()
            }
        except Exception as e:
            self.logger.debug(f"Recovery stats error: {e}")
            return {'recovered': 0, 'failed': 0, 'recovery_enabled': False}
    
    def is_available(self) -> bool:
        """Check if native C++ extension is available"""
        return NATIVE_AVAILABLE and self.manager is not None