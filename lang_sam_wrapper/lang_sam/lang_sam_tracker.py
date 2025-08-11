import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union

from lang_sam.models.coordinator import ModelCoordinator
from lang_sam.models.utils import DEVICE
from lang_sam.exceptions import LangSAMError, ErrorHandler


class LangSAMTracker:
    """Language Segment-Anything + CSRT統合トラッカー    
    責任:
    - 高レベルなAPI提供
    - モデルコーディネーターへの委譲
    - 設定管理
    
    複雑な実装詳細は各専門クラスに委譲:
    - ModelCoordinator: モデル統合・パイプライン実行
    - TrackingManager: CSRT追跡管理
    - SafeImageProcessor: 画像処理・検証
    """
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        """LangSAMTracker初期化"""
        self.sam_type = sam_type
        
        # モデルコーディネーター初期化（主要な複雑ロジックを委譲）
        self.coordinator = ModelCoordinator(sam_type, ckpt_path, device)
        
        # デフォルト追跡設定
        self._default_tracking_targets = ["white line", "red pylon", "human", "car"]
        self._setup_default_tracking()
    
    def _setup_default_tracking(self) -> None:
        """デフォルトトラッキング設定"""
        default_config = {
            'bbox_margin': 5,
            'bbox_min_size': 20,
            'tracker_min_size': 10
        }
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
        """統合推論パイプライン（GroundingDINO → CSRT → SAM2）
        
        複雑な実装詳細はModelCoordinatorに委譲
        """
        try:
            return self.coordinator.predict_full_pipeline(
                images_pil, texts_prompt, box_threshold, text_threshold,
                update_trackers, run_sam
            )
        except LangSAMError:
            # カスタム例外は再発生
            raise
        except Exception as e:
            # 一般例外をラップ
            raise ErrorHandler.handle_model_error("full_pipeline", e)
    
    def update_trackers_only(self, image_np: np.ndarray) -> Dict:
        """CSRT追跡のみ実行（高速版）
        
        トラッキング詳細はTrackingManagerに委譲
        """
        try:
            return self.coordinator.update_tracking_only(image_np)
        except LangSAMError:
            raise
        except Exception as e:
            raise ErrorHandler.handle_tracking_error("batch", "update_only", e)
    
    def update_trackers_with_sam(self, image_np: np.ndarray) -> Dict:
        """CSRT追跡 + SAM2セグメンテーション統合実行
        
        複雑な統合ロジックはModelCoordinatorに委譲
        """
        try:
            return self.coordinator.update_tracking_with_sam(image_np)
        except LangSAMError:
            raise
        except Exception as e:
            raise ErrorHandler.handle_model_error("tracking_with_sam", e)
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        """追跡対象ラベル設定
        
        設定管理はTrackingManagerに委譲
        """
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_manager.set_tracking_targets(targets)
    
    def set_tracking_config(self, config: Dict[str, int]) -> None:
        """トラッキング設定更新
        
        設定管理はTrackingConfigに委譲
        """
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_config.update(**config)
    
    def clear_trackers(self) -> None:
        """全トラッカー状態クリア
        
        状態管理はModelCoordinatorに委譲
        """
        self.coordinator.clear_tracking_state()
    
    # 互換性維持のためのプロパティ
    @property
    def has_active_tracking(self) -> bool:
        """アクティブ追跡状態確認"""
        return self.coordinator.has_active_tracking()
    
    @property 
    def tracking_targets(self) -> List[str]:
        """現在の追跡対象取得"""
        if self.coordinator.tracking_manager:
            return self.coordinator.tracking_manager.tracking_targets
        return self._default_tracking_targets
    
    @property
    def config(self) -> Dict[str, int]:
        """現在のトラッキング設定取得"""
        config_obj = self.coordinator.tracking_config
        return {
            'bbox_margin': config_obj.bbox_margin,
            'bbox_min_size': config_obj.bbox_min_size,
            'tracker_min_size': config_obj.tracker_min_size
        }
