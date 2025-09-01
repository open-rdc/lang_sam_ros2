"""AIモデル統合コーディネーター"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple

from ..models.gdino import GDINO
from ..models.sam import SAM
from ..models.utils import DEVICE
import logging


class ModelCoordinator:
    """AIモデル統合コーディネーター"""
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        self.device = device
        self.logger = logging.getLogger("model_coordinator")
        
        self.sam = self._initialize_sam(sam_type, ckpt_path, device)
        self.gdino = self._initialize_gdino(device)
        
        self.tracking_manager = None
        self.tracking_config = type('TrackingConfig', (), {
            'bbox_margin': 5, 'bbox_min_size': 3, 'tracker_min_size': 3,
            'update': lambda self, **kwargs: None
        })()
    
    def _initialize_sam(self, sam_type: str, ckpt_path: Optional[str], device) -> SAM:
        sam = SAM()
        sam.build_model(sam_type, ckpt_path, device=device)
        return sam
    
    def _initialize_gdino(self, device) -> GDINO:
        gdino = GDINO()
        gdino.build_model(device=device)
        return gdino
    
    def setup_tracking(self, tracking_targets: List[str], 
                      tracking_config: Optional[Dict[str, int]] = None,
                      csrt_params: Optional[Dict] = None) -> None:
        pass  # C++ CSRTClient使用のためスタブ
    
    def predict_full_pipeline(
        self,
        images_pil: List[Image.Image],
        texts_prompt: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        update_trackers: bool = True,
        run_sam: bool = True
    ) -> List[Dict[str, Any]]:
        
        
        gdino_results = self._run_grounding_dino(
            images_pil, texts_prompt, box_threshold, text_threshold
        )
        
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        
        for idx, result in enumerate(gdino_results):
            result = self._convert_cuda_tensors(result)
            
            if update_trackers and self.tracking_manager and result.get("labels"):
                result = self._update_tracking_integration(result, np.asarray(images_pil[idx]))
            
            processed_result = self._prepare_sam_input(result)
            
            if run_sam and processed_result.get("labels"):
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)
            
            all_results.append(processed_result)
        
        if run_sam and sam_images:
            self._run_sam_batch_inference(all_results, sam_images, sam_boxes, sam_indices)
        
        return all_results
    
    def _run_grounding_dino(
        self, images_pil: List[Image.Image], texts_prompt: List[str],
        box_threshold: float, text_threshold: float
    ) -> List[Dict[str, Any]]:
        return self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
    
    def _convert_cuda_tensors(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v.cpu().numpy() if hasattr(v, "cpu") else v) 
            for k, v in result.items()
        }
    
    def _update_tracking_integration(self, gdino_result: Dict[str, Any], 
                                   image_np: np.ndarray) -> Dict[str, Any]:
        return gdino_result
    
    def _prepare_sam_input(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **result,
            "masks": [],
            "mask_scores": [],
        }
    
    def _run_sam_batch_inference(self, all_results: List[Dict[str, Any]], 
                               sam_images: List[np.ndarray],
                               sam_boxes: List[np.ndarray], 
                               sam_indices: List[int]) -> None:
        masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
        
        for i, (idx, mask, score) in enumerate(zip(sam_indices, masks, mask_scores)):
            all_results[idx].update({
                "masks": mask,
                "mask_scores": score,
            })
    
    def update_tracking_only(self, image_np: np.ndarray) -> Dict[str, Any]:
        return self._empty_result()
    
    def update_tracking_with_sam(self, image_np: np.ndarray) -> Dict[str, Any]:
        return self._empty_result()
    
    def _process_sam_results(self, masks: Any, mask_scores: Any, 
                           num_boxes: int) -> Tuple[List, np.ndarray]:
        result_masks = []
        if (masks and isinstance(masks, (list, tuple)) and len(masks) > 0 and
            isinstance(masks[0], (list, tuple, np.ndarray)) and len(masks[0]) > 0):
            result_masks = masks[0]
        
        result_scores = np.ones(num_boxes)
        if (mask_scores and isinstance(mask_scores, (list, tuple)) and len(mask_scores) > 0 and
            isinstance(mask_scores[0], (list, tuple, np.ndarray)) and len(mask_scores[0]) > 0):
            result_scores = np.array(mask_scores[0])
        
        return result_masks, result_scores
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "boxes": np.array([]),
            "labels": [],
            "scores": np.array([]),
            "masks": [],
            "mask_scores": []
        }
    
    def clear_tracking_state(self) -> None:
        pass
    
    def has_active_tracking(self) -> bool:
        return False