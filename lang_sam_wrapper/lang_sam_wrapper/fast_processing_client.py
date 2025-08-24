#!/usr/bin/env python3
"""
Fast Processing Client - 簡素化されたC++画像処理ラッパー（同期処理のみ）
"""

import numpy as np
import cv2
from typing import List, Tuple
import logging

try:
    from lang_sam_wrapper import fast_processing
    FAST_PROCESSING_AVAILABLE = True
except ImportError:
    FAST_PROCESSING_AVAILABLE = False
    logging.warning("C++ fast_processing module not available, falling back to OpenCV")


class FastProcessingClient:
    """シンプルなC++画像処理クライアント（同期処理のみ）"""
    
    def __init__(self):
        """初期化"""
        if FAST_PROCESSING_AVAILABLE:
            self._image_processor = fast_processing.FastImageProcessor()
            logging.info("Fast processing initialized (sync only)")
        else:
            self._image_processor = None
            logging.info("Using OpenCV fallback for image processing")
    
    def bgr_to_rgb_cached(self, image: np.ndarray) -> np.ndarray:
        """BGRからRGBへの変換（C++キャッシュまたはOpenCVフォールバック）"""
        if self._image_processor is not None:
            return self._image_processor.bgr_to_rgb_cached(image)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def rgb_to_bgr_cached(self, image: np.ndarray) -> np.ndarray:
        """RGBからBGRへの変換（C++キャッシュまたはOpenCVフォールバック）"""
        if self._image_processor is not None:
            return self._image_processor.rgb_to_bgr_cached(image)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    def get_cache_size(self) -> int:
        """キャッシュサイズ取得"""
        if self._image_processor is not None:
            return self._image_processor.get_cache_size()
        return 0
    
    def clear_cache(self):
        """キャッシュクリア"""
        if self._image_processor is not None:
            self._image_processor.clear_cache()


# グローバルクライアントインスタンス（シングルトンパターン）
_fast_processing_client = None

def get_fast_processing_client() -> FastProcessingClient:
    """グローバル高速処理クライアント取得"""
    global _fast_processing_client
    if _fast_processing_client is None:
        _fast_processing_client = FastProcessingClient()
    return _fast_processing_client