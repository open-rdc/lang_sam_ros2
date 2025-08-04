import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam.models.utils import DEVICE


class LangSAMTracker:
    """LangSAM + CSRTトラッカー統合版"""
    
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type
        
        # LangSAMモデル初期化
        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path, device=device)
        self.gdino = GDINO()
        self.gdino.build_model(device=device)
        
        # CSRTトラッカー状態
        self.trackers: Dict[str, cv2.TrackerCSRT] = {}
        self.tracked_boxes: Dict[str, List[float]] = {}
        self.tracking_targets: List[str] = ["white line", "red pylon", "human", "car"]
    
    def predict_with_tracking(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        update_trackers: bool = True,
        run_sam: bool = True
    ):
        """GroundingDINO → CSRT → SAM2 統合予測"""
        
        # ステップ1: GroundingDINO検出
        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        
        for idx, result in enumerate(gdino_results):
            # CUDA tensor → CPU numpy変換
            result = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in result.items()}
            
            # ステップ2: CSRTトラッキング（オプション）
            tracked_result = result.copy()
            if update_trackers and result["labels"]:
                tracked_result = self._update_tracking(result, np.asarray(images_pil[idx]))
            
            # 最終結果準備
            processed_result = {
                **tracked_result,
                "masks": [],
                "mask_scores": [],
            }
            
            # ステップ3: SAM2用データ準備
            if run_sam and tracked_result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)
            
            all_results.append(processed_result)
        
        # ステップ3: SAM2実行（追跡結果使用）
        if run_sam and sam_images:
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update({
                    "masks": mask,
                    "mask_scores": score,
                })
        
        return all_results
    
    def _update_tracking(self, gdino_result: dict, image_np: np.ndarray) -> dict:
        """CSRTトラッキング更新"""
        try:
            boxes = gdino_result["boxes"]
            labels = gdino_result["labels"]
            scores = gdino_result.get("scores", [1.0] * len(boxes))
            
            if len(boxes) == 0:
                return gdino_result
            
            # 新しい検出結果でトラッカー初期化
            self._init_trackers(boxes, labels, image_np)
            
            # 既存トラッカー更新
            self._update_existing_trackers(image_np)
            
            # 追跡結果を返す
            if self.tracked_boxes:
                tracked_boxes = list(self.tracked_boxes.values())
                tracked_labels = list(self.tracked_boxes.keys())
                tracked_scores = [1.0] * len(tracked_boxes)
                
                return {
                    "boxes": np.array(tracked_boxes),
                    "labels": tracked_labels,
                    "scores": np.array(tracked_scores)
                }
            else:
                return gdino_result
                
        except Exception as e:
            print(f"トラッキング更新エラー: {e}")
            return gdino_result
    
    def _init_trackers(self, boxes: np.ndarray, labels: List[str], image_np: np.ndarray):
        """CSRTトラッカー初期化"""
        try:
            # 既存トラッカークリア
            self.trackers.clear()
            self.tracked_boxes.clear()
            
            height, width = image_np.shape[:2]
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                # トラッキング対象チェック
                if not any(target in label.lower() for target in self.tracking_targets):
                    continue
                
                # 座標変換（正規化 → ピクセル）
                if len(box) >= 4:
                    if np.max(box) <= 1.0:  # 正規化座標
                        x1 = int(box[0] * width)
                        y1 = int(box[1] * height)
                        x2 = int(box[2] * width)
                        y2 = int(box[3] * height)
                    else:  # ピクセル座標
                        x1, y1, x2, y2 = [int(v) for v in box]
                else:
                    continue
                
                # 境界調整
                margin = 5
                min_size = 20
                x1 = max(margin, min(x1, width - margin - min_size))
                y1 = max(margin, min(y1, height - margin - min_size))
                x2 = max(x1 + min_size, min(x2, width - margin))
                y2 = max(y1 + min_size, min(y2, height - margin))
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                if bbox_width < min_size or bbox_height < min_size:
                    continue
                
                # CSRTトラッカー初期化
                tracker = cv2.TrackerCSRT_create()
                tracker_bbox = (x1, y1, bbox_width, bbox_height)
                
                if tracker.init(image_np, tracker_bbox):
                    tracker_label = f"{label}_{i}"
                    self.trackers[tracker_label] = tracker
                    self.tracked_boxes[tracker_label] = [x1, y1, x2, y2]
                    
        except Exception as e:
            print(f"トラッカー初期化エラー: {e}")
    
    def _update_existing_trackers(self, image_np: np.ndarray):
        """既存トラッカー更新"""
        if not self.trackers:
            return
        
        labels_to_remove = []
        height, width = image_np.shape[:2]
        
        for label, tracker in self.trackers.items():
            try:
                success, bbox = tracker.update(image_np)
                
                if success and bbox is not None:
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                    
                    # 境界・サイズチェック
                    if (0 <= x1 < width and 0 <= y1 < height and 
                        x2 <= width and y2 <= height and w > 20 and h > 20):
                        self.tracked_boxes[label] = [x1, y1, x2, y2]
                    else:
                        labels_to_remove.append(label)
                else:
                    labels_to_remove.append(label)
                    
            except Exception:
                labels_to_remove.append(label)
        
        # 失敗したトラッカー削除
        for label in labels_to_remove:
            if label in self.trackers:
                del self.trackers[label]
            if label in self.tracked_boxes:
                del self.tracked_boxes[label]
    
    def update_trackers_only(self, image_np: np.ndarray) -> dict:
        """トラッカーのみ更新（検出なし）"""
        self._update_existing_trackers(image_np)
        
        if self.tracked_boxes:
            tracked_boxes = list(self.tracked_boxes.values())
            tracked_labels = list(self.tracked_boxes.keys())
            tracked_scores = [1.0] * len(tracked_boxes)
            
            return {
                "boxes": np.array(tracked_boxes),
                "labels": tracked_labels,
                "scores": np.array(tracked_scores),
                "masks": [],
                "mask_scores": []
            }
        else:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([]),
                "masks": [],
                "mask_scores": []
            }
    
    def set_tracking_targets(self, targets: List[str]):
        """トラッキング対象設定"""
        self.tracking_targets = targets
    
    def clear_trackers(self):
        """全トラッカークリア"""
        self.trackers.clear()
        self.tracked_boxes.clear()