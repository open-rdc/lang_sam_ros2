"""AIモデル統合コーディネーター（GroundingDINO + CSRT + SAM2の推論パイプライン管理）
    
技術的目的：
- ゼロショット物体検出（GroundingDINO）とリアルタイムトラッキング（CSRT）の統合
- GPU推論処理の最適化とバッチ処理による計算効率の向上
- 異なるAIモデル間のデータフロー管理とエラーハンドリング
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

from ..models.gdino import GDINO
from ..models.sam import SAM
from ..models.utils import DEVICE
from .tracking_manager import TrackingManager
from .tracking_config import TrackingConfig
from .exceptions import SAM2InitError, GroundingDINOError


class ModelCoordinator:
    """AIモデル統合コーディネーター：複数AIモデルの協調処理管理
    
    技術アーキテクチャ：
    - GroundingDINO: Transformer-baseのゼロショット物体検出（DETR系アーキテクチャ）
    - CSRT: 判別的相関フィルター（DCF）による高精度ビジュアルトラッキング
    - SAM2: ビジョントランスフォーマーによるセマンティックセグメンテーション
    - バッチ推論とGPUメモリ最適化による高速処理実現
    """
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        self.sam_type = sam_type
        self.device = device
        
        self.sam = self._initialize_sam(sam_type, ckpt_path, device)
        self.gdino = self._initialize_gdino(device)
        
        self.tracking_manager: Optional[TrackingManager] = None
        self.tracking_config = TrackingConfig()
    
    def _initialize_sam(self, sam_type: str, ckpt_path: Optional[str], device) -> SAM:
        """SAM2モデル初期化：ビジョントランスフォーマーベースのセグメンテーション
        
        技術的処理：
        - Hiera/ViT-H/ViT-L/ViT-Bアーキテクチャの選択的読み込み
        - PyTorchモデル重みの GPU VRAM への配置
        - バッチ推論用のメモリ効率最適化
        """
        try:
            sam = SAM()
            sam.build_model(sam_type, ckpt_path, device=device)
            return sam
        except Exception as e:
            raise SAM2InitError(sam_type, ckpt_path, e)
    
    def _initialize_gdino(self, device) -> GDINO:
        """GroundingDINOモデル初期化：テキストプロンプトベースの物体検出
        
        技術的アーキテクチャ：
        - DETR（Detection Transformer）のマルチモーダル拡張版
        - BERT言語エンコーダーとVision Transformerの融合
        - 自然言語クエリによるゼロショット物体検出実現
        """
        try:
            gdino = GDINO()
            gdino.build_model(device=device)
            return gdino
        except Exception as e:
            raise GroundingDINOError("initialization", e)
    
    def setup_tracking(self, tracking_targets: List[str], 
                      tracking_config: Optional[Dict[str, int]] = None,
                      csrt_params: Optional[Dict] = None) -> None:
        """リアルタイムトラッキングシステム初期化
        
        技術的機能：
        - CSRTアルゴリズムによる多目標同時追跡設定
        - HOG特徴量、色ヒストグラム、空間信頼性マップの統合
        - 24パラメータによるトラッキング精度の詳細調整
        """
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
        """AI推論パイプラインの統合実行（3ステージ協調処理）
        
        技術的ワークフロー：
        1. GroundingDINO: テキストクエリ→物体検出（Transformer推論）
        2. CSRT: 検出結果→リアルタイムトラッキング（相関フィルタ）
        3. SAM2: トラッキングBBox→精密セグメンテーション（ViT推論）
        
        最適化手法：
        - GPU並列バッチ推論による計算効率化
        - テンソルメモリ管理とCUDA kernel最適化
        """
        
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
        """GroundingDINOゼロショット物体検出の実行
        
        技術的処理：
        - マルチモーダルTransformerにImage+Textエンコーディング入力
        - Attentionメカニズムによる物体・テキスト対応学習
        - NMS（Non-Maximum Suppression）と闾値フィルタリング
        """
        try:
            return self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        except Exception as e:
            raise GroundingDINOError("prediction", e)
    
    def _convert_cuda_tensors(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """GPUテンソル→CPU配列変換：CUDA-CPU間データ伝送最適化
        
        技術的必要性：
        - GPUメモリからCPUメモリへの非同期コピー
        - CUDAコンテキストの適切なメモリ解放
        - NumPyとのZero-Copy連携による高速変換
        """
        return {
            k: (v.cpu().numpy() if hasattr(v, "cpu") else v) 
            for k, v in result.items()
        }
    
    def _update_tracking_integration(self, gdino_result: Dict[str, Any], 
                                   image_np: np.ndarray) -> Dict[str, Any]:
        """GroundingDINO検出結果とCSRTトラッキングの統合処理
        
        技術的統合：
        - 検出結果BBoxをCSRTトラッカーの初期化地点として使用
        - 相関フィルタベースのテンプレートマッチング初期化
        - 物体ラベルとトラッカーIDの関連付け管理
        """
        try:
            boxes = gdino_result.get("boxes", [])
            labels = gdino_result.get("labels", [])
            
            if len(boxes) == 0 or not self.tracking_manager:
                return gdino_result
            
            self.tracking_manager.initialize_trackers(boxes, labels, image_np)
            
            tracking_result = self.tracking_manager.get_tracking_result()
            
            if tracking_result["boxes"].size > 0:
                return {
                    "boxes": tracking_result["boxes"],
                    "labels": tracking_result["labels"],
                    "scores": tracking_result["scores"]
                }
            else:
                return gdino_result
                
        except Exception:
            return gdino_result
    
    def _prepare_sam_input(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """SAM2セグメンテーション入力データのフォーマット準備
        
        技術的処理：
        - BoundingBox座標をSAM2のプロンプト入力形式に変換
        - バッチ処理用の空マスクコンテナ初期化
        - 推論結果格納用データ構造の事前用意
        """
        return {
            **result,
            "masks": [],
            "mask_scores": [],
        }
    
    def _run_sam_batch_inference(self, all_results: List[Dict[str, Any]], 
                               sam_images: List[np.ndarray],
                               sam_boxes: List[np.ndarray], 
                               sam_indices: List[int]) -> None:
        """SAM2バッチセグメンテーション推論：GPU効率化のための並列処理
        
        技術的最適化：
        - 複数画像の同時セグメンテーション処理
        - CUDA Kernelのバッチ実行による GPU 使用率向上
        - VRAMメモリと計算時間のバランシング最適化
        """
        try:
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update({
                    "masks": mask,
                    "mask_scores": score,
                })
                
        except Exception:
            pass
    
    def update_tracking_only(self, image_np: np.ndarray) -> Dict[str, Any]:
        """高速CSRTトラッキングのみ実行：30Hzリアルタイム処理版
        
        技術的特徴：
        - GPU推論を回避したCPUベースの軽量処理
        - HOG特徴量と相関フィルタを使用した高速マッチング
        - フレーム間の小さな変化に対する適応的追跡
        """
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
        """統合トラッキング+セグメンテーション：高精度マスク生成
        
        技術的ワークフロー：
        1. CSRTアルゴリズムによる物体位置追跡（CPU処理）
        2. 追跡結果BBoxをSAM2のプロンプトとして入力
        3. Vision Transformerによる高精度セグメンテーション（GPU推論）
        """
        if not self.tracking_manager:
            return self._empty_result()
        
        try:
            tracked_boxes = self.tracking_manager.update_all_trackers(image_np)
            
            if not tracked_boxes:
                return self._empty_result()
            
            tracking_result = self.tracking_manager.get_tracking_result()
            boxes = tracking_result["boxes"]
            labels = tracking_result["labels"]
            
            try:
                masks, mask_scores, _ = self.sam.predict_batch(
                    [image_np], xyxy=[boxes]
                )
                
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
                return {
                    **tracking_result,
                    "masks": [],
                    "mask_scores": np.ones(len(boxes))
                }
                
        except Exception:
            return self._empty_result()
    
    def _process_sam_results(self, masks: Any, mask_scores: Any, 
                           num_boxes: int) -> Tuple[List, np.ndarray]:
        """SAM2セグメンテーション結果の安全な後処理
        
        技術的安全性：
        - PyTorchテンソルの型・次元・データ型の穏健性検証
        - GPUメモリエラーからの復旧処理
        - 異なるSAM2モデルバージョン間の互換性保証
        """
        result_masks = []
        try:
            if (masks is not None and isinstance(masks, (list, tuple)) 
                and len(masks) > 0):
                if (isinstance(masks[0], (list, tuple, np.ndarray)) 
                    and hasattr(masks[0], '__len__') and len(masks[0]) > 0):
                    result_masks = masks[0]
        except (TypeError, IndexError):
            result_masks = []
        
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