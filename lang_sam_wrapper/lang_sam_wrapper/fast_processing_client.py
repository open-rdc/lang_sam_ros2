#!/usr/bin/env python3
"""
Fast Processing Client - C++高速化機能のPythonラッパー
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
import logging

from lang_sam_wrapper import fast_processing
FAST_PROCESSING_AVAILABLE = True


class FastProcessingClient:
    """C++高速化処理のPythonクライアント"""
    
    def __init__(self, num_async_workers: int = 3, max_queue_size: int = 10):
        """初期化
        
        Args:
            num_async_workers: 非同期処理ワーカー数
            max_queue_size: 最大キューサイズ
        """
        self._image_processor = fast_processing.FastImageProcessor()
        self._async_processor = fast_processing.AsyncProcessor(
            num_async_workers, max_queue_size, max_queue_size * 2
        )
        logging.info(f"Fast processing initialized: {num_async_workers} workers")
    
    def bgr_to_rgb_cached(self, image: np.ndarray) -> np.ndarray:
        """BGRからRGBへの変換（キャッシュあり）"""
        return self._image_processor.bgr_to_rgb_cached(image)
    
    def rgb_to_bgr_cached(self, image: np.ndarray) -> np.ndarray:
        """RGBからBGRへの変換（キャッシュあり）"""
        return self._image_processor.rgb_to_bgr_cached(image)
    
    def convert_detections_fast(self, detections: List[List[int]]) -> List[Tuple[int, int, int, int]]:
        """検出結果の高速変換"""
        rects = self._image_processor.convert_detections_fast(detections)
        return [(rect.x, rect.y, rect.width, rect.height) for rect in rects]
    
    def draw_boxes_fast(self, image: np.ndarray, 
                       boxes: List[Tuple[int, int, int, int]],
                       labels: List[str] = None,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """高速バウンディングボックス描画"""
        # C++のBoundingBoxオブジェクトに変換
        cpp_boxes = []
        labels = labels or []
        
        for i, (x, y, w, h) in enumerate(boxes):
            label = labels[i] if i < len(labels) else ""
            cpp_box = fast_processing.BoundingBox(x, y, w, h, label, 1.0)
            cpp_boxes.append(cpp_box)
        
        cv_color = (color[2], color[1], color[0])  # BGR変換
        return self._image_processor.draw_boxes_fast(image, cpp_boxes, cv_color, thickness)
    
    def submit_grounding_dino_task_async(self, frame_id: int, image: np.ndarray,
                                        text_prompt: str, box_threshold: float, text_threshold: float) -> bool:
        """GroundingDINO検出を非同期で投入"""
        return self._async_processor.submit_grounding_dino_task(
            frame_id, image, text_prompt, box_threshold, text_threshold
        )
    
    def submit_sam2_task_async(self, frame_id: int, image: np.ndarray,
                              boxes: List[Tuple[int, int, int, int]],
                              labels: List[str]) -> bool:
        """SAM2処理を非同期で投入"""
        # Tupleリストに変換
        box_tuples = [(x, y, w, h) for x, y, w, h in boxes]
        return self._async_processor.submit_sam2_task(frame_id, image, box_tuples, labels)
    
    def submit_visualization_task_async(self, frame_id: int, image: np.ndarray,
                                       boxes: List[Tuple[int, int, int, int]],
                                       labels: List[str],
                                       output_topic: str) -> bool:
        """可視化処理を非同期で投入"""
        box_tuples = [(x, y, w, h) for x, y, w, h in boxes]
        return self._async_processor.submit_visualization_task(
            frame_id, image, box_tuples, labels, output_topic
        )
    
    def get_async_result(self, frame_id: int, task_type: str) -> Optional[np.ndarray]:
        """非同期処理結果を取得"""
        result = self._async_processor.get_result(frame_id, task_type)
        if result and result.success:
            return result.result_image
        elif result and not result.success:
            logging.error(f"Async task failed: {result.error_message}")
        return None
    
    def has_async_result(self, frame_id: int, task_type: str) -> bool:
        """非同期処理結果が利用可能かチェック"""
        return self._async_processor.has_result(frame_id, task_type)
    
    def get_cache_size(self) -> int:
        """キャッシュサイズを取得"""
        return self._image_processor.get_cache_size()
    
    def get_async_stats(self) -> dict:
        """非同期処理の統計情報を取得"""
        return {
            "available": True,
            "queue_size": self._async_processor.get_queue_size(),
            "results_size": self._async_processor.get_results_size(),
            "active_workers": self._async_processor.get_active_workers(),
            "cache_size": self.get_cache_size()
        }
    
    def clear_all_cache(self):
        """すべてのキャッシュをクリア"""
        self._image_processor.clear_all_cache()
        self._async_processor.clear_all_results()
    
    def shutdown(self):
        """リソースを解放"""
        self._async_processor.shutdown()
        logging.info("Fast processing client shutdown completed")


# Global instance for easy access
_global_fast_client = None

def get_fast_processing_client() -> FastProcessingClient:
    """グローバルFastProcessingClientインスタンスを取得"""
    global _global_fast_client
    if _global_fast_client is None:
        _global_fast_client = FastProcessingClient()
    return _global_fast_client

def shutdown_fast_processing():
    """グローバルFastProcessingClientをシャットダウン"""
    global _global_fast_client
    if _global_fast_client is not None:
        _global_fast_client.shutdown()
        _global_fast_client = None