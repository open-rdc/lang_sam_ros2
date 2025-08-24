"""CSRTマルチターゲットトラッキングマネージャ：複数物体同時追跡システム

技術的機能：
- CSRT（Channel and Spatial Reliability Tracking）アルゴリズムの複数インスタンス管理
- 相関フィルタベースの高精度物体追跡と失敗検出
- フレーム間連続性とリアルタイム性の両立
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .tracking_config import TrackingConfig
from .csrt_tracker import CSRTTracker
from .exceptions import CSRTTrackingError


class TrackingManager:
    """複数物体同時追跡システム：CSRTアルゴリズムのスケーラブル管理
    
    技術的アーキテクチャ：
    - 相関フィルタ（DCF: Discriminative Correlation Filter）ベースの高精度追跡
    - HOG特徴量 + 色ヒストグラムのマルチモーダル特徴量統合
    - スケール適応とモデルアップデートの動的最適化
    - 24パラメータの細かなチューニング対応
    
    パフォーマンス特性：
    - CPUベースで高速動作（GPUメモリ不要）
    - フレームレートに対する線形スケーリング特性
    """
    
    def __init__(self, tracking_targets: List[str], config: Optional[TrackingConfig] = None,
                 csrt_params: Optional[Dict] = None):
        self.tracking_targets = [target.lower() for target in tracking_targets]
        self.config = config or TrackingConfig()
        self.csrt_params = csrt_params or {}
        self.trackers: Dict[str, CSRTTracker] = {}
        self.tracked_boxes: Dict[str, List[float]] = {}
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        """追跡対象更新"""
        self.tracking_targets = [target.lower() for target in targets]
    
    def set_config(self, config: TrackingConfig) -> None:
        """設定更新"""
        self.config = config
    
    def initialize_trackers(self, boxes: np.ndarray, labels: List[str], image: np.ndarray) -> None:
        """複数トラッカー初期化（ラベル順序保持）"""
        try:
            self.clear_trackers()
            
            height, width = image.shape[:2]
            valid_tracker_count = 0
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if not self._is_target_label(label):
                    continue
                
                bbox = self._process_bbox(box, width, height)
                if bbox is None:
                    continue
                
                # 有効なトラッカー番号を使用（インデックス不一致を防止）
                tracker_id = f"{label}_{valid_tracker_count}"
                try:
                    tracker = CSRTTracker(tracker_id, bbox, image, self.csrt_params)
                    if tracker.is_initialized:
                        self.trackers[tracker_id] = tracker
                        self.tracked_boxes[tracker_id] = list(bbox)
                        valid_tracker_count += 1
                except CSRTTrackingError:
                    continue
                    
        except Exception as e:
            raise CSRTTrackingError("batch", "initialization", e)
    
    def _is_target_label(self, label: str) -> bool:
        """追跡対象判定（完全一致または接頭詞一致）"""
        label_lower = label.lower()
        # 完全一致を優先
        for target in self.tracking_targets:
            target_lower = target.lower()
            if label_lower == target_lower or label_lower.startswith(target_lower):
                return True
        return False
    
    def _process_bbox(self, box: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """バウンディングボックス座標の正規化とロバスト検証"""
        try:
            if len(box) < 4:
                return None
            
            if np.max(box) <= 1.0:
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
            else:
                x1, y1, x2, y2 = [int(v) for v in box]
            
            bbox = self._adjust_bbox_bounds(x1, y1, x2, y2, width, height)
            
            if self._validate_bbox_size(bbox):
                return bbox
            return None
            
        except Exception:
            return None
    
    def _adjust_bbox_bounds(self, x1: int, y1: int, x2: int, y2: int, 
                           width: int, height: int) -> Tuple[int, int, int, int]:
        """BoundingBox境界調整"""
        margin = self.config.bbox_margin
        min_size = self.config.bbox_min_size
        
        x1 = max(margin, min(x1, width - margin - min_size))
        y1 = max(margin, min(y1, height - margin - min_size))
        x2 = max(x1 + min_size, min(x2, width - margin))
        y2 = max(y1 + min_size, min(y2, height - margin))
        
        return (x1, y1, x2, y2)
    
    def _validate_bbox_size(self, bbox: Tuple[int, int, int, int]) -> bool:
        """BoundingBoxサイズ検証"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width >= self.config.bbox_min_size and height >= self.config.bbox_min_size
    
    def update_all_trackers(self, image: np.ndarray) -> Dict[str, List[float]]:
        """全CSRTトラッカーの一括更新と失敗管理"""
        updated_boxes = {}
        failed_trackers = []
        
        for tracker_id, tracker in self.trackers.items():
            bbox = tracker.update(image)
            if bbox is not None and tracker.is_tracking():
                updated_boxes[tracker_id] = list(bbox)
                self.tracked_boxes[tracker_id] = list(bbox)
            else:
                failed_trackers.append(tracker_id)
        
        # 失敗したトラッカーを削除
        for tracker_id in failed_trackers:
            del self.trackers[tracker_id]
            if tracker_id in self.tracked_boxes:
                del self.tracked_boxes[tracker_id]
        
        return updated_boxes
    
    def clear_trackers(self) -> None:
        """全トラッカークリア"""
        self.trackers.clear()
        self.tracked_boxes.clear()
    
    def get_active_trackers(self) -> Dict[str, List[float]]:
        """アクティブトラッカー取得"""
        return self.tracked_boxes.copy()
    
    def get_tracker_count(self) -> int:
        """トラッカー数取得"""
        return len(self.trackers)