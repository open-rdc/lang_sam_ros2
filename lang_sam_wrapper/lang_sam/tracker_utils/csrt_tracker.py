"""単一オブジェクト用CSRTトラッカー"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any

from .exceptions import CSRTTrackingError


class CSRTTracker:
    """単一オブジェクト用CSRTトラッカー（パラメータ化対応）"""
    
    def __init__(self, tracker_id: str, bbox: Tuple[int, int, int, int], 
                 image: np.ndarray, csrt_params: Optional[Dict] = None):
        self.tracker_id = tracker_id
        self.csrt_params = csrt_params or {}
        self.tracker = self._create_tracker()
        self.is_initialized = False
        
        if self.tracker and self._initialize(image, bbox):
            self.is_initialized = True
    
    def _create_tracker(self) -> Optional[cv2.Tracker]:
        """OpenCV CSRTトラッカー生成（パラメータ設定付き）"""
        try:
            tracker = self._create_opencv_tracker()
            self._apply_csrt_params(tracker)
            return tracker
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "tracker creation", e)
    
    def _create_opencv_tracker(self) -> cv2.Tracker:
        """OpenCVバージョン対応のCSRTトラッカー生成"""
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()
    
    def _apply_csrt_params(self, tracker: cv2.Tracker) -> None:
        """CSRTパラメータをトラッカーに適用"""
        if tracker and self.csrt_params and hasattr(tracker, 'setParams'):
            params = self._build_csrt_params()
            if params:
                tracker.setParams(params)
    
    def _build_csrt_params(self) -> Any:
        """CSRTパラメータ構築"""
        try:
            params = self._create_opencv_params()
            if not params:
                return None
            
            self._set_csrt_parameters(params)
            return params
        except Exception:
            return None
    
    def _create_opencv_params(self) -> Any:
        """OpenCVパラメータ構造体作成"""
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_Params'):
            return cv2.legacy.TrackerCSRT_Params()
        elif hasattr(cv2, 'TrackerCSRT_Params'):
            return cv2.TrackerCSRT_Params()
        return None
    
    def _set_csrt_parameters(self, params: Any) -> None:
        """CSRTパラメータ値設定"""
        param_mapping = self._get_param_mapping()
        
        for param_key, param_value in self.csrt_params.items():
            if param_key.startswith('csrt_'):
                clean_key = param_key[5:]
                opencv_attr = param_mapping.get(clean_key)
                if opencv_attr and hasattr(params, opencv_attr):
                    setattr(params, opencv_attr, param_value)
    
    def _get_param_mapping(self) -> Dict[str, str]:
        """CSRTパラメータマッピング取得"""
        return {
            'use_hog': 'useHog', 'use_color_names': 'useColorNames', 
            'use_gray': 'useGray', 'use_rgb': 'useRGB',
            'use_channel_weights': 'useChannelWeights', 'use_segmentation': 'useSegmentation',
            'window_function': 'windowFunction', 'kaiser_alpha': 'kaiserAlpha',
            'cheb_attenuation': 'chebAttenuation', 'template_size': 'templateSize',
            'gsl_sigma': 'gslSigma', 'hog_orientations': 'hogOrientations',
            'hog_clip': 'hogClip', 'padding': 'padding',
            'filter_lr': 'filterLR', 'weights_lr': 'weightsLR',
            'num_hog_channels_used': 'numHogChannelsUsed', 'admm_iterations': 'admmIterations',
            'histogram_bins': 'histogramBins', 'histogram_lr': 'histogramLR',
            'background_ratio': 'backgroundRatio', 'number_of_scales': 'numberOfScales',
            'scale_sigma_factor': 'scaleSigmaFactor', 'scale_model_max_area': 'scaleModelMaxArea'
        }
    
    def _initialize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """トラッカー初期化"""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            tracker_bbox = (x1, y1, width, height)
            
            bgr_image = self._ensure_bgr_format(image)
            return self.tracker.init(bgr_image, tracker_bbox)
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "initialization", e)
    
    def _ensure_bgr_format(self, image: np.ndarray) -> np.ndarray:
        """BGR形式確保"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    def update(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """追跡更新
        
        Returns:
            Optional[Tuple]: 更新されたbbox座標 (x1, y1, x2, y2) or None
        """
        if not self.is_initialized:
            return None
        
        try:
            bgr_image = self._ensure_bgr_format(image)
            success, bbox = self.tracker.update(bgr_image)
            
            if success and bbox is not None:
                x1, y1, w, h = [int(v) for v in bbox]
                return (x1, y1, x1 + w, y1 + h)
            return None
            
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "update", e)