import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam.models.utils import DEVICE


class LangSAMTracker:
    """Language Segment-Anything + CSRT統合トラッカー
    
    統合処理フロー:
    1. GroundingDINO: テキストプロンプト → ゼロショット物体検出
    2. CSRT: Channel and Spatial Reliability Tracking → リアルタイム追跡
    3. SAM2: Segment Anything Model 2 → 高精度セグメンテーション
    """
    
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type
        
        # LangSAMモデル群初期化（GPU推論用）
        self.sam = SAM()  # SAM2セグメンテーションモデル
        self.sam.build_model(sam_type, ckpt_path, device=device)
        self.gdino = GDINO()  # GroundingDINOゼロショット検出モデル
        self.gdino.build_model(device=device)
        
        # CSRT物体追跡状態管理
        self.trackers: Dict[str, cv2.TrackerCSRT] = {}  # OpenCVトラッカー辞書
        self.tracked_boxes: Dict[str, List[float]] = {}  # 追跡中BoundingBox座標
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
        """GroundingDINO → CSRT → SAM2 統合推論パイプライン"""
        
        # ステップ1: GroundingDINOによるテキストプロンプト検出
        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        
        for idx, result in enumerate(gdino_results):
            # CUDA tensor → CPU numpy変換（GPU推論結果処理）
            result = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in result.items()}
            
            # ステップ2: CSRT追跡統合（オプション）
            tracked_result = result.copy()
            if update_trackers and result["labels"]:
                tracked_result = self._update_tracking(result, np.asarray(images_pil[idx]))
            
            # SAM2用結果準備
            processed_result = {
                **tracked_result,
                "masks": [],
                "mask_scores": [],
            }
            
            # ステップ3: SAM2用画像・BOXデータ準備
            if run_sam and tracked_result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)
            
            all_results.append(processed_result)
        
        # ステップ3: SAM2バッチ推論（追跡結果ベース）
        if run_sam and sam_images:
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update({
                    "masks": mask,
                    "mask_scores": score,
                })
        
        return all_results
    
    def _update_tracking(self, gdino_result: dict, image_np: np.ndarray) -> dict:
        """CSRT追跡状態更新（GroundingDINO検出結果ベース）"""
        try:
            boxes = gdino_result["boxes"]
            labels = gdino_result["labels"]
            scores = gdino_result.get("scores", [1.0] * len(boxes))
            
            if len(boxes) == 0:
                return gdino_result
            
            # GroundingDINO結果によるCSRTトラッカー再初期化
            self._init_trackers(boxes, labels, image_np)
            
            # 既存CSRTトラッカー状態更新
            self._update_existing_trackers(image_np)
            
            # 追跡結果返却
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
        """CSRTトラッカー初期化（OpenCV 4.11.0対応版）"""
        try:
            # 既存トラッカー状態クリア
            self.trackers = {}
            self.tracked_boxes = {}
            
            height, width = image_np.shape[:2]
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                # 追跡対象フィルタリング
                if not any(target in label.lower() for target in self.tracking_targets):
                    continue
                
                # 座標系変換（正規化座標 → ピクセル座標）
                if len(box) >= 4:
                    if np.max(box) <= 1.0:  # 正規化座標系
                        x1 = int(box[0] * width)
                        y1 = int(box[1] * height)
                        x2 = int(box[2] * width)
                        y2 = int(box[3] * height)
                    else:  # ピクセル座標系
                        x1, y1, x2, y2 = [int(v) for v in box]
                else:
                    continue
                
                # BoundingBox境界調整（画像サイズ制約）
                margin = 5
                min_size = 20
                x1 = max(margin, min(x1, width - margin - min_size))
                y1 = max(margin, min(y1, height - margin - min_size))
                x2 = max(x1 + min_size, min(x2, width - margin))
                y2 = max(y1 + min_size, min(y2, height - margin))
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # 最小サイズフィルタ
                if bbox_width < min_size or bbox_height < min_size:
                    continue
                
                # OpenCV CSRTトラッカー生成（4.11.0 legacy API対応）
                try:
                    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    else:
                        tracker = cv2.TrackerCSRT_create()
                except Exception:
                    continue
                
                # BGR画像でトラッカー初期化（OpenCV標準色空間）
                tracker_bbox = (x1, y1, bbox_width, bbox_height)
                init_image = image_np if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                
                try:
                    if tracker.init(init_image, tracker_bbox):
                        tracker_label = f"{label}_{i}"
                        self.trackers[tracker_label] = tracker
                        self.tracked_boxes[tracker_label] = [x1, y1, x2, y2]
                except Exception:
                    pass
                    
        except Exception:
            pass
    
    def _update_existing_trackers(self, image_np: np.ndarray):
        """CSRT追跡更新（BBOXクリッピング対応版）"""
        if not self.trackers:
            return
        
        labels_to_remove = []
        height, width = image_np.shape[:2]
        
        # BGR色空間画像使用（OpenCV標準）
        bgr_image = image_np if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # 辞書安全反復処理
        for label, tracker in list(self.trackers.items()):
            if label not in self.trackers:
                continue
                
            try:
                success, bbox = tracker.update(bgr_image)
                
                if success and bbox is not None:
                    x1, y1, w, h = [int(v) for v in bbox]
                    x2, y2 = x1 + w, y1 + h
                    
                    # BoundingBoxクリッピング（画像境界内制限）
                    x1_clipped = max(0, min(x1, width-1))
                    y1_clipped = max(0, min(y1, height-1))
                    x2_clipped = max(x1_clipped+1, min(x2, width))
                    y2_clipped = max(y1_clipped+1, min(y2, height))
                    
                    w_clipped = x2_clipped - x1_clipped
                    h_clipped = y2_clipped - y1_clipped
                    
                    # クリップ後最小サイズチェック
                    if w_clipped > 10 and h_clipped > 10:
                        self.tracked_boxes[label] = [x1_clipped, y1_clipped, x2_clipped, y2_clipped]
                    else:
                        labels_to_remove.append(label)
                else:
                    labels_to_remove.append(label)
                    
            except Exception:
                labels_to_remove.append(label)
        
        # 失敗トラッカー削除
        for label in labels_to_remove:
            self.trackers.pop(label, None)
            self.tracked_boxes.pop(label, None)
    
    def update_trackers_only(self, image_np: np.ndarray) -> dict:
        """CSRT追跡のみ実行（高速版、SAM2無し）"""
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
    
    def update_trackers_with_sam(self, image_np: np.ndarray) -> dict:
        """CSRT追跡 + SAM2セグメンテーション統合実行（毎フレーム版）"""
        # CSRT状態更新
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
        
        # 追跡対象なし時の早期リターン
        if len(tracked_boxes) == 0:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([]),
                "masks": [],
                "mask_scores": []
            }
        
        try:
            # SAM2バッチ推論（追跡BoundingBox基準）
            masks, mask_scores, _ = self.sam.predict_batch(
                [image_np], 
                xyxy=[np.array(tracked_boxes)]
            )
            
            # SAM2結果安全処理
            result_masks = []
            result_mask_scores = []
            
            # セグメンテーションマスク安全抽出
            try:
                if masks is not None and isinstance(masks, (list, tuple)) and len(masks) > 0:
                    if isinstance(masks[0], (list, tuple, np.ndarray)) and hasattr(masks[0], '__len__') and len(masks[0]) > 0:
                        result_masks = masks[0]
            except (TypeError, IndexError):
                result_masks = []
                
            # マスク信頼度スコア安全抽出
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
            # SAM2エラー時はCSRT結果のみ返却
            return {
                "boxes": np.array(tracked_boxes),
                "labels": tracked_labels,
                "scores": np.ones(len(tracked_boxes)),
                "masks": [],
                "mask_scores": np.ones(len(tracked_boxes))
            }
    
    def set_tracking_targets(self, targets: List[str]):
        """追跡対象ラベル設定"""
        self.tracking_targets = targets
    
    def clear_trackers(self):
        """全トラッカー状態クリア"""
        self.trackers = {}
        self.tracked_boxes = {}