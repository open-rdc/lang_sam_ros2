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
            print(f"=== CSRT初期化開始 ===")
            print(f"検出オブジェクト数: {len(boxes)}")
            print(f"検出ラベル: {labels}")
            print(f"トラッキング対象: {self.tracking_targets}")
            
            # 既存トラッカークリア（安全な操作）
            old_count = len(self.trackers)
            # 辞書を安全にクリア（空の辞書に置き換え）
            self.trackers = {}
            self.tracked_boxes = {}
            print(f"既存トラッカー{old_count}個をクリア")
            
            height, width = image_np.shape[:2]
            print(f"画像サイズ: {width}x{height}")
            successful_inits = 0
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                print(f"処理中 {i}: ラベル='{label}', bbox={box}")
                
                # トラッキング対象チェック
                is_target = any(target in label.lower() for target in self.tracking_targets)
                print(f"トラッキング対象判定: {is_target}")
                if not is_target:
                    print(f"スキップ: '{label}'は対象外")
                    continue
                
                # 座標変換（正規化 → ピクセル）
                if len(box) >= 4:
                    if np.max(box) <= 1.0:  # 正規化座標
                        x1 = int(box[0] * width)
                        y1 = int(box[1] * height)
                        x2 = int(box[2] * width)
                        y2 = int(box[3] * height)
                        print(f"正規化座標変換: {box} → ({x1},{y1},{x2},{y2})")
                    else:  # ピクセル座標
                        x1, y1, x2, y2 = [int(v) for v in box]
                        print(f"ピクセル座標使用: ({x1},{y1},{x2},{y2})")
                else:
                    print(f"無効なbbox形式: {box}")
                    continue
                
                # 境界調整
                margin = 5
                min_size = 20
                x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2
                x1 = max(margin, min(x1, width - margin - min_size))
                y1 = max(margin, min(y1, height - margin - min_size))
                x2 = max(x1 + min_size, min(x2, width - margin))
                y2 = max(y1 + min_size, min(y2, height - margin))
                
                if (x1_orig, y1_orig, x2_orig, y2_orig) != (x1, y1, x2, y2):
                    print(f"境界調整: ({x1_orig},{y1_orig},{x2_orig},{y2_orig}) → ({x1},{y1},{x2},{y2})")
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                print(f"bbox サイズ: {bbox_width}x{bbox_height}")
                
                if bbox_width < min_size or bbox_height < min_size:
                    print(f"サイズ不足でスキップ: {bbox_width}x{bbox_height} < {min_size}")
                    continue
                
                # OpenCV 4.11.0対応のCSRTトラッカー初期化
                try:
                    # legacy APIを優先使用（OpenCV 4.x系の互換性問題対応）
                    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        tracker = cv2.legacy.TrackerCSRT_create()
                        print(f"Legacy CSRT使用")
                    else:
                        tracker = cv2.TrackerCSRT_create()
                        print(f"標準CSRT使用")
                except Exception as e:
                    print(f"CSRTトラッカー作成失敗: {e}")
                    continue
                
                tracker_bbox = (x1, y1, bbox_width, bbox_height)
                
                # 画像形式を統一（BGR）
                init_image = image_np if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                
                # トラッカー初期化実行
                try:
                    init_result = tracker.init(init_image, tracker_bbox)
                    if init_result:
                        tracker_label = f"{label}_{i}"
                        self.trackers[tracker_label] = tracker
                        self.tracked_boxes[tracker_label] = [x1, y1, x2, y2]
                        successful_inits += 1
                        print(f"SUCCESS: '{tracker_label}' 初期化成功 bbox={tracker_bbox}")
                    else:
                        print(f"FAILED: '{label}_{i}' 初期化失敗")
                        
                except Exception as e:
                    print(f"FAILED: '{label}_{i}' 初期化例外 {e}")
            
            print(f"=== CSRT初期化完了: {successful_inits}/{len(boxes)}個成功 ===")
                    
        except Exception as e:
            print(f"CSRT初期化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_existing_trackers(self, image_np: np.ndarray):
        """最適化されたCSRTトラッカー更新（安全な辞書操作）"""
        if not self.trackers:
            print("CSRTアップデート: トラッカーなし")
            return
        
        print(f"CSRTアップデート: {len(self.trackers)}個のトラッカーを更新")
        labels_to_remove = []
        height, width = image_np.shape[:2]
        successful_updates = 0
        
        # BGR画像使用（一貫性保持）
        bgr_image = image_np if len(image_np.shape) == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # 安全な反復処理のため辞書のコピーを作成
        tracker_items = list(self.trackers.items())
        
        for label, tracker in tracker_items:
            # トラッカーが削除済みかチェック
            if label not in self.trackers:
                continue
                
            try:
                success, bbox = tracker.update(bgr_image)
                
                if success and bbox is not None:
                    x1, y1, w, h = [int(v) for v in bbox]
                    x2, y2 = x1 + w, y1 + h
                    
                    # 境界チェック
                    if (0 <= x1 < width and 0 <= y1 < height and 
                        x2 <= width and y2 <= height and w > 15 and h > 15):
                        self.tracked_boxes[label] = [x1, y1, x2, y2]
                        successful_updates += 1
                        print(f"SUCCESS: '{label}' → ({x1},{y1},{x2},{y2})")
                    else:
                        print(f"FAIL: '{label}' 境界外 ({x1},{y1},{x2},{y2}) in {width}x{height}")
                        labels_to_remove.append(label)
                else:
                    print(f"FAIL: '{label}' トラッキング失敗 success={success}, bbox={bbox}")
                    labels_to_remove.append(label)
                    
            except Exception as e:
                print(f"FAIL: '{label}' 例外発生 {e}")
                labels_to_remove.append(label)
        
        # 失敗したトラッカー削除（安全に実行）
        if labels_to_remove:
            print(f"削除: {labels_to_remove}")
            for label in labels_to_remove:
                self.trackers.pop(label, None)
                self.tracked_boxes.pop(label, None)
        
        print(f"CSRTアップデート完了: {successful_updates}/{len(tracker_items)}個成功")
    
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
            
        except Exception as e:
            print(f"SAM2処理エラー: {e}")
            return {
                "boxes": np.array(tracked_boxes),
                "labels": tracked_labels,
                "scores": np.ones(len(tracked_boxes)),
                "masks": [],
                "mask_scores": np.ones(len(tracked_boxes))  # デフォルトスコア
            }