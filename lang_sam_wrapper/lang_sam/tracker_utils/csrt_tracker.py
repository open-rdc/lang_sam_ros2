"""単一オブジェクト用CSRTトラッカー"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any

from .exceptions import CSRTTrackingError

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # CSRTパラメータログを表示するためINFOレベルに設定


class CSRTTracker:
    """単一オブジェクト用CSRTトラッカー（シンプル実装）"""
    
    def __init__(self, tracker_id: str, bbox: Tuple[int, int, int, int], 
                 image: np.ndarray, csrt_params: Optional[Dict] = None):
        self.tracker_id = tracker_id
        self.csrt_params = csrt_params or {}
        
        # 強制ログ出力 - CSRTパラメータ確認
        print(f"[CSRT_DEBUG] {tracker_id}: csrt_params = {self.csrt_params}")
        logger.info(f"[{self.tracker_id}] CSRTパラメータ受け取り: {len(self.csrt_params)}個")
        
        logger.info(f"[{self.tracker_id}] CSRT tracker初期化開始 - bbox: {bbox}, image shape: {image.shape}")
        logger.info(f"[{self.tracker_id}] CSRTパラメータ詳細: {self.csrt_params}")
        
        # OpenCV CSRTトラッカー初期化
        self.tracker = None
        self.is_initialized = False
        self.lost_frames_count = 0
        self.last_bbox = bbox
        
        try:
            self._create_csrt_tracker()
            # バウンディングボックスの初期化
            self.is_initialized = self.tracker.init(image, bbox)
            
            if self.is_initialized:
                logger.info(f"[{self.tracker_id}] ✅ CSRT tracker created with config.yaml parameters")
            else:
                logger.error(f"[{self.tracker_id}] ❌ Failed to initialize CSRT tracker")
                raise CSRTTrackingError(self.tracker_id, "initialization", "Failed to initialize tracker")
                
        except Exception as e:
            logger.error(f"[{self.tracker_id}] ❌ CSRT tracker initialization error: {e}")
            self.is_initialized = False
            raise CSRTTrackingError(self.tracker_id, "initialization", str(e))
    
    def _create_csrt_tracker(self):
        """OpenCV CSRTトラッカーの作成（パラメータ付き）"""
        try:
            # OpenCV 4.11.0対応: パラメータ設定の試行
            try:
                # 新しいAPI（OpenCV 4.5+）を試行
                params = cv2.TrackerCSRT.Params()
                
                # config.yamlからのパラメータを適用
                self._apply_csrt_parameters(params)
                
                # パラメータ付きトラッカー作成を試行
                self.tracker = cv2.TrackerCSRT.create(params)
                logger.info(f"[{self.tracker_id}] Created CSRT tracker with parameters")
                
            except (AttributeError, TypeError) as e:
                logger.warning(f"[{self.tracker_id}] Parameter-based creation failed: {e}")
                # フォールバック: パラメータなしトラッカー
                self.tracker = cv2.TrackerCSRT.create()
                logger.info(f"[{self.tracker_id}] Created CSRT tracker without parameters (fallback)")
                
        except Exception as e:
            # OpenCV 4.5以下のレガシーAPI
            try:
                self.tracker = cv2.legacy.TrackerCSRT_create()
                logger.info(f"[{self.tracker_id}] Created CSRT tracker using legacy API")
            except AttributeError:
                raise CSRTTrackingError(self.tracker_id, "creation", f"Cannot create CSRT tracker: {e}")
    
    def _apply_csrt_parameters(self, params):
        """CSRTパラメータの適用"""
        if not self.csrt_params:
            logger.info(f"[{self.tracker_id}] No CSRT parameters provided, using defaults")
            return
        
        # パラメータマッピング（config.yamlのキー → OpenCVパラメータ属性）
        param_mapping = {
            'csrt_use_hog': 'use_hog',
            'csrt_use_color_names': 'use_color_names', 
            'csrt_use_gray': 'use_gray',
            'csrt_use_rgb': 'use_rgb',
            'csrt_use_channel_weights': 'use_channel_weights',
            'csrt_use_segmentation': 'use_segmentation',
            'csrt_window_function': 'window_function',
            'csrt_kaiser_alpha': 'kaiser_alpha',
            'csrt_cheb_attenuation': 'cheb_attenuation',
            'csrt_template_size': 'template_size',
            'csrt_gsl_sigma': 'gsl_sigma',
            'csrt_hog_orientations': 'hog_orientations',
            'csrt_hog_clip': 'hog_clip',
            'csrt_padding': 'padding',
            'csrt_filter_lr': 'filter_lr',
            'csrt_weights_lr': 'weights_lr',
            'csrt_num_hog_channels_used': 'num_hog_channels_used',
            'csrt_admm_iterations': 'admm_iterations',
            'csrt_histogram_bins': 'histogram_bins',
            'csrt_histogram_lr': 'histogram_lr',
            'csrt_background_ratio': 'background_ratio',
            'csrt_number_of_scales': 'number_of_scales',
            'csrt_scale_sigma_factor': 'scale_sigma_factor',
            'csrt_scale_model_max_area': 'scale_model_max_area',
            'csrt_scale_lr': 'scale_lr',
            'csrt_scale_step': 'scale_step'
        }
        
        applied_params = []
        
        for config_key, param_attr in param_mapping.items():
            if config_key in self.csrt_params:
                value = self.csrt_params[config_key]
                try:
                    if hasattr(params, param_attr):
                        setattr(params, param_attr, value)
                        applied_params.append(f"{param_attr}={value}")
                    else:
                        logger.warning(f"[{self.tracker_id}] Parameter {param_attr} not found in OpenCV CSRT params")
                except Exception as e:
                    logger.warning(f"[{self.tracker_id}] Failed to set {param_attr}={value}: {e}")
        
        if applied_params:
            logger.info(f"[{self.tracker_id}] 適用されたCSRTパラメータ: {', '.join(applied_params)}")
        else:
            logger.info(f"[{self.tracker_id}] CSRTパラメータ適用なし")
    
    def update(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """トラッキング更新"""
        if not self.is_initialized or self.tracker is None:
            return None
        
        try:
            # OpenCV 4.11.0対応: update()がNoneを返す場合の処理
            result = self.tracker.update(image)
            
            # OpenCV 4.11.0では、update()がNoneを返すことがある
            if result is None:
                logger.warning(f"[{self.tracker_id}] Update returned None (OpenCV 4.11.0 issue)")
                # 例外ベースの検証を試行
                try:
                    # Trackerが有効かテスト
                    test_bbox = self.tracker.update(image)
                    if test_bbox is not None and len(test_bbox) == 4:
                        success = True
                        bbox = test_bbox
                    else:
                        success = False
                        bbox = None
                except:
                    success = False
                    bbox = None
            else:
                success, bbox = result
            
            if success and bbox is not None:
                self.last_bbox = tuple(int(x) for x in bbox)
                self.lost_frames_count = 0
                return self.last_bbox
            else:
                self.lost_frames_count += 1
                return None
                
        except Exception as e:
            logger.error(f"[{self.tracker_id}] Update error: {e}")
            self.lost_frames_count += 1
            return None
    
    def is_tracking(self) -> bool:
        """トラッキング状態確認"""
        return self.is_initialized and self.lost_frames_count < 10  # 10フレーム以上失敗で無効
    
    def reset(self, bbox: Tuple[int, int, int, int], image: np.ndarray) -> bool:
        """トラッカーリセット"""
        try:
            self._create_csrt_tracker()
            self.is_initialized = self.tracker.init(image, bbox)
            if self.is_initialized:
                self.last_bbox = bbox
                self.lost_frames_count = 0
                logger.info(f"[{self.tracker_id}] Tracker reset successfully")
            return self.is_initialized
        except Exception as e:
            logger.error(f"[{self.tracker_id}] Reset error: {e}")
            return False