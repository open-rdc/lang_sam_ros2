"""モデル統合コーディネーター - SAM2, GroundingDINO, CSRT統合管理"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam.models.utils import DEVICE
from lang_sam.lang_sam_tracker import TrackingManager, TrackingConfig, ModelError, SAM2InitError, GroundingDINOError, ErrorHandler


class ModelCoordinator:
    """AIモデル統合コーディネーター"""
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        self.sam_type = sam_type
        self.device = device
        
        # モデル初期化
        self.sam = self._initialize_sam(sam_type, ckpt_path, device)
        self.gdino = self._initialize_gdino(device)
        
        # トラッキングマネージャー初期化
        self.tracking_manager: Optional[TrackingManager] = None
        self.tracking_config = TrackingConfig()
    
    def _initialize_sam(self, sam_type: str, ckpt_path: Optional[str], device) -> SAM:
        """SAM2モデル初期化"""
        try:
            sam = SAM()
            sam.build_model(sam_type, ckpt_path, device=device)
            return sam
        except Exception as e:
            raise SAM2InitError(sam_type, ckpt_path, e)
    
    def _initialize_gdino(self, device) -> GDINO:
        """GroundingDINOモデル初期化"""
        try:
            gdino = GDINO()
            gdino.build_model(device=device)
            return gdino
        except Exception as e:
            raise GroundingDINOError("initialization", e)
    
    def setup_tracking(self, tracking_targets: List[str], 
                      tracking_config: Optional[Dict[str, int]] = None,
                      csrt_params: Optional[Dict] = None) -> None:
        """トラッキング設定（CSRTパラメータ対応）"""
        if tracking_config:
            self.tracking_config.update(**tracking_config)
        
        self.tracking_manager = TrackingManager(tracking_targets, self.tracking_config, csrt_params)
    
    def predict_full_pipeline(
        self,
        images_pil: List[Image.Image],
        texts_prompt: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        update_trackers: bool = True,
        run_sam: bool = True
    ) -> List[Dict[str, Any]]:
        """完全パイプライン実行（GroundingDINO → CSRT → SAM2）"""
        
        # ステップ1: GroundingDINO検出
        gdino_results = self._run_grounding_dino(
            images_pil, texts_prompt, box_threshold, text_threshold
        )
        
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        
        for idx, result in enumerate(gdino_results):
            # CUDA tensor → CPU numpy変換
            result = self._convert_cuda_tensors(result)
            
            # ステップ2: CSRT追跡統合
            if update_trackers and self.tracking_manager and result.get("labels"):
                result = self._update_tracking_integration(result, np.asarray(images_pil[idx]))
            
            # SAM2用データ準備
            processed_result = self._prepare_sam_input(result)
            
            # SAM2バッチ処理用データ収集
            if run_sam and processed_result.get("labels"):
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)
            
            all_results.append(processed_result)
        
        # ステップ3: SAM2バッチ推論
        if run_sam and sam_images:
            self._run_sam_batch_inference(all_results, sam_images, sam_boxes, sam_indices)
        
        return all_results
    
    def _run_grounding_dino(
        self, images_pil: List[Image.Image], texts_prompt: List[str],
        box_threshold: float, text_threshold: float
    ) -> List[Dict[str, Any]]:
        """GroundingDINO検出実行"""
        try:
            return self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        except Exception as e:
            raise GroundingDINOError("prediction", e)
    
    def _convert_cuda_tensors(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """CUDA tensor → CPU numpy変換"""
        return {
            k: (v.cpu().numpy() if hasattr(v, "cpu") else v) 
            for k, v in result.items()
        }
    
    def _update_tracking_integration(self, gdino_result: Dict[str, Any], 
                                   image_np: np.ndarray) -> Dict[str, Any]:
        """CSRT追跡統合"""
        try:
            boxes = gdino_result.get("boxes", [])
            labels = gdino_result.get("labels", [])
            
            if len(boxes) == 0 or not self.tracking_manager:
                return gdino_result
            
            # トラッカー再初期化
            self.tracking_manager.initialize_trackers(boxes, labels, image_np)
            
            # 追跡結果取得
            tracking_result = self.tracking_manager.get_tracking_result()
            
            if tracking_result["boxes"].size > 0:
                return {
                    "boxes": tracking_result["boxes"],
                    "labels": tracking_result["labels"],
                    "scores": tracking_result["scores"]
                }
            else:
                return gdino_result
                
        except Exception as e:
            # トラッキングエラー時はGroundingDINO結果を使用
            return gdino_result
    
    def _prepare_sam_input(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """SAM2入力用データ準備"""
        return {
            **result,
            "masks": [],
            "mask_scores": [],
        }
    
    def _run_sam_batch_inference(self, all_results: List[Dict[str, Any]], 
                               sam_images: List[np.ndarray],
                               sam_boxes: List[np.ndarray], 
                               sam_indices: List[int]) -> None:
        """SAM2バッチ推論実行"""
        try:
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update({
                    "masks": mask,
                    "mask_scores": score,
                })
                
        except Exception as e:
            # SAM2エラー時は元の結果を保持
            pass
    
    def update_tracking_only(self, image_np: np.ndarray) -> Dict[str, Any]:
        """CSRT追跡のみ実行（高速モード）"""
        if not self.tracking_manager:
            return self._empty_result()
        
        try:
            tracked_boxes = self.tracking_manager.update_all_trackers(image_np)
            if tracked_boxes:
                return self.tracking_manager.get_tracking_result()
            else:
                return self._empty_result()
                
        except Exception:
            return self._empty_result()
    
    def update_tracking_with_sam(self, image_np: np.ndarray) -> Dict[str, Any]:
        """CSRT追跡 + SAM2セグメンテーション実行"""
        if not self.tracking_manager:
            return self._empty_result()
        
        try:
            # CSRT追跡更新
            tracked_boxes = self.tracking_manager.update_all_trackers(image_np)
            
            if not tracked_boxes:
                return self._empty_result()
            
            tracking_result = self.tracking_manager.get_tracking_result()
            boxes = tracking_result["boxes"]
            labels = tracking_result["labels"]
            
            # SAM2推論実行
            try:
                masks, mask_scores, _ = self.sam.predict_batch(
                    [image_np], xyxy=[boxes]
                )
                
                # 結果安全処理
                result_masks, result_scores = self._process_sam_results(
                    masks, mask_scores, len(boxes)
                )
                
                return {
                    "boxes": boxes,
                    "labels": labels,
                    "scores": tracking_result["scores"],
                    "masks": result_masks,
                    "mask_scores": result_scores
                }
                
            except Exception:
                # SAM2エラー時はCSRT結果のみ返却
                return {
                    **tracking_result,
                    "masks": [],
                    "mask_scores": np.ones(len(boxes))
                }
                
        except Exception:
            return self._empty_result()
    
    def _process_sam_results(self, masks: Any, mask_scores: Any, 
                           num_boxes: int) -> Tuple[List, np.ndarray]:
        """SAM2結果安全処理"""
        # マスク安全抽出
        result_masks = []
        try:
            if (masks is not None and isinstance(masks, (list, tuple)) 
                and len(masks) > 0):
                if (isinstance(masks[0], (list, tuple, np.ndarray)) 
                    and hasattr(masks[0], '__len__') and len(masks[0]) > 0):
                    result_masks = masks[0]
        except (TypeError, IndexError):
            result_masks = []
        
        # スコア安全抽出
        try:
            if (mask_scores is not None and isinstance(mask_scores, (list, tuple)) 
                and len(mask_scores) > 0):
                if (isinstance(mask_scores[0], (list, tuple, np.ndarray)) 
                    and hasattr(mask_scores[0], '__len__') and len(mask_scores[0]) > 0):
                    result_scores = np.array(mask_scores[0])
                else:
                    result_scores = np.ones(num_boxes)
            else:
                result_scores = np.ones(num_boxes)
        except (TypeError, IndexError):
            result_scores = np.ones(num_boxes)
        
        return result_masks, result_scores
    
    def _empty_result(self) -> Dict[str, Any]:
        """空の結果セット"""
        return {
            "boxes": np.array([]),
            "labels": [],
            "scores": np.array([]),
            "masks": [],
            "mask_scores": []
        }
    
    def clear_tracking_state(self) -> None:
        """追跡状態クリア"""
        if self.tracking_manager:
            self.tracking_manager.clear_trackers()
    
    def has_active_tracking(self) -> bool:
        """アクティブ追跡確認"""
        return (self.tracking_manager is not None 
                and self.tracking_manager.has_active_trackers())