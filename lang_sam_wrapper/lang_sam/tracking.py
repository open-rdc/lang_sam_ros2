"""CSRTトラッキング管理 - 単一責任原則に基づく分割"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from lang_sam.exceptions import CSRTTrackingError, ErrorHandler


class TrackingConfig:
    """トラッキング設定管理"""
    
    def __init__(self, bbox_margin: int = 5, bbox_min_size: int = 20, 
                 tracker_min_size: int = 10):
        self.bbox_margin = bbox_margin
        self.bbox_min_size = bbox_min_size
        self.tracker_min_size = tracker_min_size
    
    def update(self, **kwargs) -> None:
        """設定更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class CSRTTracker:
    """単一オブジェクト用CSRTトラッカー"""
    
    def __init__(self, tracker_id: str, bbox: Tuple[int, int, int, int], 
                 image: np.ndarray):
        self.tracker_id = tracker_id
        self.tracker = self._create_tracker()
        self.is_initialized = False
        
        if self.tracker and self._initialize(image, bbox):
            self.is_initialized = True
    
    def _create_tracker(self) -> Optional[cv2.Tracker]:
        """OpenCV CSRTトラッカー生成（バージョン対応）"""
        try:
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                return cv2.legacy.TrackerCSRT_create()
            else:
                return cv2.TrackerCSRT_create()
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "tracker creation", e)
    
    def _initialize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """トラッカー初期化"""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            tracker_bbox = (x1, y1, width, height)
            
            # BGR画像でトラッカー初期化
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


class TrackingManager:
    """複数オブジェクトCSRTトラッキング管理"""
    
    def __init__(self, tracking_targets: List[str], config: Optional[TrackingConfig] = None):
        self.tracking_targets = [target.lower() for target in tracking_targets]
        self.config = config or TrackingConfig()
        self.trackers: Dict[str, CSRTTracker] = {}
        self.tracked_boxes: Dict[str, List[float]] = {}
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        """追跡対象更新"""
        self.tracking_targets = [target.lower() for target in targets]
    
    def set_config(self, config: TrackingConfig) -> None:
        """設定更新"""
        self.config = config
    
    def initialize_trackers(self, boxes: np.ndarray, labels: List[str], image: np.ndarray) -> None:
        """複数トラッカー初期化"""
        try:
            # 既存トラッカー状態クリア
            self.clear_trackers()
            
            height, width = image.shape[:2]
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                # 追跡対象フィルタリング
                if not self._is_target_label(label):
                    continue
                
                # 座標変換・検証
                bbox = self._process_bbox(box, width, height)
                if bbox is None:
                    continue
                
                # トラッカー生成
                tracker_id = f"{label}_{i}"
                try:
                    tracker = CSRTTracker(tracker_id, bbox, image)
                    if tracker.is_initialized:
                        self.trackers[tracker_id] = tracker
                        self.tracked_boxes[tracker_id] = list(bbox)
                except CSRTTrackingError:
                    # 個別トラッカー初期化失敗は継続
                    continue
                    
        except Exception as e:
            raise CSRTTrackingError("batch", "initialization", e)
    
    def _is_target_label(self, label: str) -> bool:
        """追跡対象判定"""
        label_lower = label.lower()
        return any(target in label_lower for target in self.tracking_targets)
    
    def _process_bbox(self, box: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """BoundingBox座標処理・検証"""
        try:
            if len(box) < 4:
                return None
            
            # 正規化座標→ピクセル座標変換
            if np.max(box) <= 1.0:  # 正規化座標系
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
            else:  # ピクセル座標系
                x1, y1, x2, y2 = [int(v) for v in box]
            
            # 境界調整
            bbox = self._adjust_bbox_bounds(x1, y1, x2, y2, width, height)
            
            # サイズ検証
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
        """全トラッカー更新"""
        if not self.trackers:
            return {}
        
        height, width = image.shape[:2]
        failed_trackers = []
        
        for tracker_id, tracker in list(self.trackers.items()):
            try:
                updated_bbox = tracker.update(image)
                
                if updated_bbox is not None:
                    # クリッピング処理
                    clipped_bbox = self._clip_bbox_to_image(updated_bbox, width, height)
                    
                    if self._validate_clipped_bbox(clipped_bbox):
                        self.tracked_boxes[tracker_id] = list(clipped_bbox)
                    else:
                        failed_trackers.append(tracker_id)
                else:
                    failed_trackers.append(tracker_id)
                    
            except CSRTTrackingError:
                failed_trackers.append(tracker_id)
        
        # 失敗トラッカー削除
        for tracker_id in failed_trackers:
            self._remove_tracker(tracker_id)
        
        return self.tracked_boxes.copy()
    
    def _clip_bbox_to_image(self, bbox: Tuple[int, int, int, int], 
                           width: int, height: int) -> Tuple[int, int, int, int]:
        """画像境界内BoundingBoxクリッピング"""
        x1, y1, x2, y2 = bbox
        x1_clipped = max(0, min(x1, width - 1))
        y1_clipped = max(0, min(y1, height - 1))
        x2_clipped = max(x1_clipped + 1, min(x2, width))
        y2_clipped = max(y1_clipped + 1, min(y2, height))
        return (x1_clipped, y1_clipped, x2_clipped, y2_clipped)
    
    def _validate_clipped_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """クリップ後BoundingBox検証"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width > self.config.tracker_min_size and height > self.config.tracker_min_size
    
    def _remove_tracker(self, tracker_id: str) -> None:
        """トラッカー削除"""
        self.trackers.pop(tracker_id, None)
        self.tracked_boxes.pop(tracker_id, None)
    
    def clear_trackers(self) -> None:
        """全トラッカー状態クリア"""
        self.trackers.clear()
        self.tracked_boxes.clear()
    
    def get_tracking_result(self) -> Dict[str, Any]:
        """追跡結果取得"""
        if not self.tracked_boxes:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([])
            }
        
        boxes = list(self.tracked_boxes.values())
        labels = list(self.tracked_boxes.keys())
        scores = np.ones(len(boxes))
        
        return {
            "boxes": np.array(boxes),
            "labels": labels,
            "scores": scores
        }
    
    def has_active_trackers(self) -> bool:
        """アクティブトラッカー存在確認"""
        return len(self.trackers) > 0