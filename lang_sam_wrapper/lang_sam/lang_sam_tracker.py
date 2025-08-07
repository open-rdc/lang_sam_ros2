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
                
        except Exception:
            return gdino_result
    
    def _init_trackers(self, boxes: np.ndarray, labels: List[str], image_np: np.ndarray):
        """CSRTトラッカー初期化"""
        try:
            # 既存トラッカークリア
            self.trackers = {}
            self.tracked_boxes = {}
            
            height, width = image_np.shape[:2]
            successful_inits = 0
            
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
                try:
                    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    else:
                        tracker = cv2.TrackerCSRT_create()
                except Exception:
                    continue
                
                tracker_bbox = (x1, y1, bbox_width, bbox_height)
                init_image = image_np if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                
                # トラッカー初期化実行
                try:
                    if tracker.init(init_image, tracker_bbox):
                        tracker_label = f"{label}_{i}"
                        self.trackers[tracker_label] = tracker
                        self.tracked_boxes[tracker_label] = [x1, y1, x2, y2]
                        successful_inits += 1
                except Exception:
                    pass
                    
        except Exception:
            pass
    
    def _update_existing_trackers(self, image_np: np.ndarray):
        """CSRTトラッカー更新（BBOXクリッピング対応）"""
        if not self.trackers:
            return
        
        labels_to_remove = []
        height, width = image_np.shape[:2]
        
        # BGR画像使用
        bgr_image = image_np if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # 安全な反復処理
        for label, tracker in list(self.trackers.items()):
            if label not in self.trackers:
                continue
                
            try:
                success, bbox = tracker.update(bgr_image)
                
                if success and bbox is not None:
                    x1, y1, w, h = [int(v) for v in bbox]
                    x2, y2 = x1 + w, y1 + h
                    
                    # BBOXクリッピング（画像境界内に制限）
                    x1_clipped = max(0, min(x1, width-1))
                    y1_clipped = max(0, min(y1, height-1))
                    x2_clipped = max(x1_clipped+1, min(x2, width))
                    y2_clipped = max(y1_clipped+1, min(y2, height))
                    
                    w_clipped = x2_clipped - x1_clipped
                    h_clipped = y2_clipped - y1_clipped
                    
                    # 最小サイズチェック（クリップ後）
                    if w_clipped > 10 and h_clipped > 10:
                        self.tracked_boxes[label] = [x1_clipped, y1_clipped, x2_clipped, y2_clipped]
                    else:
                        labels_to_remove.append(label)
                else:
                    labels_to_remove.append(label)
                    
            except Exception:
                labels_to_remove.append(label)
        
        # 失敗したトラッカー削除
        for label in labels_to_remove:
            self.trackers.pop(label, None)
            self.tracked_boxes.pop(label, None)
    
    def update_trackers_only(self, image_np: np.ndarray) -> dict:
        """CSRT tracking only (fast)"""
        self._update_existing_trackers(image_np)
        
        if not self.tracked_boxes:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([]),
                "masks": [],
                "mask_scores": []
            }
            
        return {
            "boxes": np.array(list(self.tracked_boxes.values())),
            "labels": list(self.tracked_boxes.keys()),
            "scores": np.ones(len(self.tracked_boxes)),
            "masks": [],
            "mask_scores": []
        }
    
    def set_tracking_targets(self, targets: List[str]):
        """トラッキング対象設定"""
        self.tracking_targets = targets
    
    def clear_trackers(self):
        """全トラッカークリア（安全な操作）"""
        self.trackers = {}
        self.tracked_boxes = {}
    
    def update_trackers_with_sam(self, image_np: np.ndarray) -> dict:
        """最適化されたCSRT + SAM2（毎フレーム実行）"""
        # Update CSRT trackers
        self._update_existing_trackers(image_np)
        
        if not self.tracked_boxes:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([]),
                "masks": [],
                "mask_scores": []
            }
        
        tracked_boxes = list(self.tracked_boxes.values())
        tracked_labels = list(self.tracked_boxes.keys())
        
        # Skip SAM2 if no tracked boxes
        if len(tracked_boxes) == 0:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([]),
                "masks": [],
                "mask_scores": []
            }
        
        try:
            # Fast SAM2 prediction on valid boxes
            masks, mask_scores, _ = self.sam.predict_batch(
                [image_np], 
                xyxy=[np.array(tracked_boxes)]
            )
            
            # SAM2結果の安全な処理
            result_masks = []
            result_mask_scores = []
            
            # masksの安全な処理
            try:
                if masks is not None and isinstance(masks, (list, tuple)) and len(masks) > 0:
                    if isinstance(masks[0], (list, tuple, np.ndarray)) and hasattr(masks[0], '__len__') and len(masks[0]) > 0:
                        result_masks = masks[0]
            except (TypeError, IndexError):
                result_masks = []
                
            # mask_scoresの安全な処理  
            try:
                if mask_scores is not None and isinstance(mask_scores, (list, tuple)) and len(mask_scores) > 0:
                    if isinstance(mask_scores[0], (list, tuple, np.ndarray)) and hasattr(mask_scores[0], '__len__') and len(mask_scores[0]) > 0:
                        result_mask_scores = mask_scores[0]
                    else:
                        result_mask_scores = np.ones(len(tracked_boxes))
                else:
                    result_mask_scores = np.ones(len(tracked_boxes))
            except (TypeError, IndexError):
                result_mask_scores = np.ones(len(tracked_boxes))
            
            return {
                "boxes": np.array(tracked_boxes),
                "labels": tracked_labels,
                "scores": np.ones(len(tracked_boxes)),
                "masks": result_masks,
                "mask_scores": result_mask_scores
            }
            
        except Exception:
            return {
                "boxes": np.array(tracked_boxes),
                "labels": tracked_labels,
                "scores": np.ones(len(tracked_boxes)),
                "masks": [],
                "mask_scores": np.ones(len(tracked_boxes))  # デフォルトスコア
            }