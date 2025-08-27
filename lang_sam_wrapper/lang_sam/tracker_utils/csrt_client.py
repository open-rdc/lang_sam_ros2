#!/usr/bin/env python3
"""
Native C++ CSRT Tracker Client with ROS Parameter Integration
C++実装のCSRTトラッカーへのPythonインターフェースを提供するクライアント
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import numpy as np
import cv2

try:
    # C++拡張モジュールのインポート
    # pybind11でビルドされたC++モジュールをPythonから利用する目的で使用
    from lang_sam_wrapper.csrt_native import CSRTParams, CSRTTrackerNative, CSRTManagerNative
    NATIVE_AVAILABLE = True
    print("[CSRTClient] C++ CSRT extension loaded successfully")
except ImportError as e:
    print(f"[CSRTClient] C++ extension not available: {e}")
    NATIVE_AVAILABLE = False

class CSRTClient:
    """C++ CSRTトラッカークライアント
    
    技術的役割：
    - pybind11経由でC++実装のCSRTトラッカーをPythonから操作する目的で使用
    - ROS2パラメータシステムから27個のCSRTパラメータを動的に読み込む目的で使用
    - リアルタイム性能を実現するためのC++/Pythonハイブリッドアーキテクチャ
    """
    
    def __init__(self, node: Node):
        self.node = node
        self.logger = node.get_logger()
        # ラベルキャッシュ：C++側への無駄なメソッド呼び出しを削減する目的で使用
        self._cached_labels = []
        
        if not NATIVE_AVAILABLE:
            self.logger.error("C++ CSRT extension not available")
            self.manager = None
            return
            
        # ROSパラメータからCSRT設定を初期化
        # config.yamlから動的にパラメータを読み込みC++側に伝達する目的で使用
        self.params = self._load_params_from_ros()
        # C++ CSRTマネージャー初期化：複数トラッカーを統合管理する目的で使用
        self.manager = CSRTManagerNative(self.params)
        
        # Log parameter loading
        self._log_parameters()
        
        self.logger.info("CSRTNativeClient initialized with native C++ CSRT")
    
    def _load_params_from_ros(self) -> 'CSRTParams':
        """ROSパラメータからCSRT設定をロード
        
        config.yamlの27個のCSRTパラメータをC++構造体にマッピングする目的で使用。
        各パラメータはトラッキング精度と速度のトレードオフを制御。
        """
        params = CSRTParams()  # C++側のCSRTパラメータ構造体
        
        # パラメータマッピング定義：ROSパラメータ名とC++構造体フィールドの対応
        # 各パラメータの技術的意味：
        param_mapping = {
            'csrt_use_hog': (params.use_hog, bool),           # HOG特徴量を使用してエッジ情報を追跡する目的で使用
            'csrt_use_color_names': (params.use_color_names, bool),  # 色名特徴で色情報を活用する目的で使用
            'csrt_use_gray': (params.use_gray, bool),         # グレースケール変換で輝度情報のみを使用する目的で使用
            'csrt_use_rgb': (params.use_rgb, bool),           # RGBカラー情報を保持して追跡精度を向上させる目的で使用
            'csrt_use_channel_weights': (params.use_channel_weights, bool),
            'csrt_use_segmentation': (params.use_segmentation, bool),
            'csrt_window_function': (params.window_function, str),
            'csrt_kaiser_alpha': (params.kaiser_alpha, float),
            'csrt_cheb_attenuation': (params.cheb_attenuation, float),
            'csrt_template_size': (params.template_size, float),     # テンプレートサイズ：大きいほど精度向上但し計算量増加
            'csrt_gsl_sigma': (params.gsl_sigma, float),             # ガウシアンカーネルの幅：探索範囲を制御する目的で使用
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
            'csrt_psr_threshold': (params.psr_threshold, float),     # PSR闾値：低いほど追跡継続しやすくする目的で使用
        }
        
        for param_name, (default_value, param_type) in param_mapping.items():
            try:
                # Declare parameter
                self.node.declare_parameter(param_name, default_value)
                # Get parameter value
                value = self.node.get_parameter(param_name).value
                
                # CSRTParams構造体への値設定
                # プレフィックス'csrt_'を除去してC++構造体フィールド名に変換する目的で使用
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
    
    
    def is_available(self) -> bool:
        """Check if native C++ extension is available"""
        return NATIVE_AVAILABLE and self.manager is not None