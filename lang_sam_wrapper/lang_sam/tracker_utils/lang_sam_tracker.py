"""Language-SAM統合トラッカー：高レベルAPI"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional

from ..models.utils import DEVICE
from .model_coordinator import ModelCoordinator
# カスタム例外削除 - 標準Exception使用


class LangSAMTracker:
    """Language Segment-Anything統合システムの高レベルAPI"""
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        self.sam_type = sam_type
        
        self.coordinator = ModelCoordinator(sam_type, ckpt_path, device)
        
        self._default_tracking_targets = ["white line", "red pylon", "human", "car"]
        self._setup_default_tracking()
    
    def _setup_default_tracking(self) -> None:
        default_config = {'bbox_margin': 5, 'bbox_min_size': 20, 'tracker_min_size': 10}
        self.coordinator.setup_tracking(self._default_tracking_targets, default_config)
    
    def predict_with_tracking(
        self,
        images_pil: List[Image.Image],
        texts_prompt: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        update_trackers: bool = True,
        run_sam: bool = True
    ) -> List[Dict]:
        return self.coordinator.predict_full_pipeline(
            images_pil, texts_prompt, box_threshold, text_threshold,
            update_trackers, run_sam
        )
    
    def update_trackers_only(self, image_np: np.ndarray) -> Dict:
        return self.coordinator.update_tracking_only(image_np)
    
    def update_trackers_with_sam(self, image_np: np.ndarray) -> Dict:
        return self.coordinator.update_tracking_with_sam(image_np)
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_manager.set_tracking_targets(targets)
    
    def set_tracking_config(self, config: Dict[str, int]) -> None:
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_config.update(**config)
    
    def set_csrt_params(self, csrt_params: Dict) -> None:
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_manager.csrt_params = csrt_params
        else:
            default_config = {
                'bbox_margin': 5,
                'bbox_min_size': 20,
                'tracker_min_size': 10
            }
            self.coordinator.setup_tracking(self._default_tracking_targets, default_config, csrt_params)
    
    def clear_trackers(self) -> None:
        self.coordinator.clear_tracking_state()
    
    @property
    def has_active_tracking(self) -> bool:
        return self.coordinator.has_active_tracking()
    
    @property 
    def tracking_targets(self) -> List[str]:
        if self.coordinator.tracking_manager:
            return self.coordinator.tracking_manager.tracking_targets
        return self._default_tracking_targets
    
    @property
    def config(self) -> Dict[str, int]:
        config_obj = self.coordinator.tracking_config
        return {
            'bbox_margin': config_obj.bbox_margin,
            'bbox_min_size': config_obj.bbox_min_size,
            'tracker_min_size': config_obj.tracker_min_size
        }