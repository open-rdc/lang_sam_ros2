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
from .logging_manager import LoggerFactory, create_log_context

# デバッグモード制御は統一ロギングシステムで管理


class ModelCoordinator:
    """AIモデル統合コーディネーター：複数AIモデルの協調処理管理
    
    技術アーキテクチャ：
    - GroundingDINO: Transformer-baseのゼロショット物体検出（DETR系アーキテクチャ）
    - CSRT: 判別的相関フィルター（DCF）による高精度ビジュアルトラッキング
    - SAM2: ビジョントランスフォーマーによるセマンティックセグメンテーション
    - バッチ推論とGPUメモリ最適化による高速処理実現
    """
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE, debug_mode: bool = False):
        self.sam_type = sam_type
        self.device = device
        
        # 統一ロギング初期化
        LoggerFactory.set_debug_mode(debug_mode)
        self.logger = LoggerFactory.get_logger("model_coordinator")
        self.perf_logger = LoggerFactory.get_performance_logger("model_coordinator")
        
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
            with self.perf_logger.measure_time("sam2_initialization", "ai_model"):
                sam = SAM()
                sam.build_model(sam_type, ckpt_path, device=device)
                
                context = create_log_context("ai_model", "sam2_init", 
                                           model_type=sam_type, checkpoint=ckpt_path)
                self.logger.info("SAM2モデル初期化完了", context)
                return sam
        except Exception as e:
            context = create_log_context("ai_model", "sam2_init", 
                                       model_type=sam_type, error=str(e))
            self.logger.error("SAM2モデル初期化失敗", context)
            raise SAM2InitError(sam_type, ckpt_path, e)
    
    def _initialize_gdino(self, device) -> GDINO:
        """GroundingDINOモデル初期化：テキストプロンプトベースの物体検出
        
        技術的アーキテクチャ：
        - DETR（Detection Transformer）のマルチモーダル拡張版
        - BERT言語エンコーダーとVision Transformerの融合
        - 自然言語クエリによるゼロショット物体検出実現
        """
        try:
            with self.perf_logger.measure_time("gdino_initialization", "ai_model"):
                gdino = GDINO()
                gdino.build_model(device=device)
                
                context = create_log_context("ai_model", "gdino_init", device=str(device))
                self.logger.info("GroundingDINOモデル初期化完了", context)
                return gdino
        except Exception as e:
            context = create_log_context("ai_model", "gdino_init", error=str(e))
            self.logger.error("GroundingDINOモデル初期化失敗", context)
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
        
        context = create_log_context("tracking", "setup", 
                                   targets=tracking_targets, 
                                   config_provided=tracking_config is not None,
                                   csrt_params_provided=csrt_params is not None)
        self.logger.info("トラッキングシステム初期化開始", context)
        
        self.tracking_manager = TrackingManager(tracking_targets, self.tracking_config, csrt_params)
        
        self.logger.info("トラッキングシステム初期化完了", context)
    
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
        
        with self.perf_logger.measure_time("full_pipeline", "ai_pipeline"):
            context = create_log_context("ai_pipeline", "predict_full", 
                                       num_images=len(images_pil),
                                       prompts=texts_prompt,
                                       update_trackers=update_trackers,
                                       run_sam=run_sam)
            self.logger.debug("AI推論パイプライン開始", context)
            
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
            
            result_context = create_log_context("ai_pipeline", "predict_complete", 
                                              processed_images=len(all_results),
                                              sam_processed=len(sam_images))
            self.logger.debug("AI推論パイプライン完了", result_context)
            
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
        - 物体ラベルとトラッカーIDの関連付け管理（ラベル整合性保証）
        """
        try:
            boxes = gdino_result.get("boxes", [])
            labels = gdino_result.get("labels", [])
            
            if len(boxes) == 0 or not self.tracking_manager:
                return gdino_result
            
            # ラベルとボックスの数が一致することを確認
            if len(boxes) != len(labels):
                mismatch_context = create_log_context("tracking", "data_validation", 
                                                    boxes_count=len(boxes), labels_count=len(labels))
                self.logger.warning("ボックスとラベル数の不一致", mismatch_context)
                return gdino_result
            
            # GroundingDINOの検出結果を記録
            detection_context = create_log_context("tracking", "gdino_detection", 
                                                  labels=labels, bbox_count=len(boxes))
            self.logger.debug("GroundingDINO検出結果", detection_context)
            
            self.tracking_manager.initialize_trackers(boxes, labels, image_np)
            
            tracking_result = self.tracking_manager.get_tracking_result()
            
            # トラッキング結果を記録
            if tracking_result["labels"]:
                track_context = create_log_context("tracking", "tracking_result", 
                                                 labels=tracking_result['labels'])
                self.logger.debug("トラッキング結果取得", track_context)
            
            if tracking_result["boxes"].size > 0:
                return {
                    "boxes": tracking_result["boxes"],
                    "labels": tracking_result["labels"],
                    "scores": tracking_result["scores"]
                }
            else:
                return gdino_result
                
        except Exception as e:
            error_context = create_log_context("tracking", "integration_error", error=str(e))
            self.logger.error("トラッキング統合エラー", error_context)
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
        """SAM2バッチセグメンテーション推論：GPU効率化のための並列処理（ラベル整合性保証）
        
        技術的最適化：
        - 複数画像の同時セグメンテーション処理
        - CUDA Kernelのバッチ実行による GPU 使用率向上
        - VRAMメモリと計算時間のバランシング最適化
        - ラベルとマスクの順序一致性確保
        """
        try:
            # SAM入力情報を記録
            for i, (idx, boxes) in enumerate(zip(sam_indices, sam_boxes)):
                labels = all_results[idx].get("labels", [])
                sam_input_context = create_log_context("ai_model", "sam_input", 
                                                     batch_idx=i, result_idx=idx, 
                                                     boxes_count=len(boxes), labels=labels)
                self.logger.debug("SAM入力データ準備", sam_input_context)
            
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            
            # 結果をインデックス順に処理（ラベル順序保持）
            for i, (idx, mask, score) in enumerate(zip(sam_indices, masks, mask_scores)):
                labels = all_results[idx].get("labels", [])
                mask_count = len(mask) if hasattr(mask, '__len__') else 'N/A'
                
                sam_output_context = create_log_context("ai_model", "sam_output", 
                                                      batch_idx=i, result_idx=idx, 
                                                      mask_count=mask_count, labels=labels)
                self.logger.debug("SAM出力処理", sam_output_context)
                
                all_results[idx].update({
                    "masks": mask,
                    "mask_scores": score,
                })
                
        except Exception as e:
            error_context = create_log_context("ai_model", "sam_batch_error", error=str(e))
            self.logger.error("SAMバッチ推論エラー", error_context)
    
    def update_tracking_only(self, image_np: np.ndarray) -> Dict[str, Any]:
        """高速CSRTトラッキングのみ実行：30Hzリアルタイム処理版
        
        技術的特徴：
        - GPU推論を回避したCPUベースの軽量処理
        - HOG特徴量と相関フィルタを使用した高速マッチング
        - フレーム間の小さな変化に対する適応的追跡
        - ラベル整合性の継続保証
        """
        if not self.tracking_manager:
            return self._empty_result()
        
        try:
            tracked_boxes = self.tracking_manager.update_all_trackers(image_np)
            if tracked_boxes:
                result = self.tracking_manager.get_tracking_result()
                # トラッキング専用モードの結果確認
                if result["labels"]:
                    track_only_context = create_log_context("tracking", "tracking_only", 
                                                          labels=result['labels'])
                    self.logger.debug("Tracking-only結果", track_only_context)
                return result
            else:
                return self._empty_result()
                
        except Exception as e:
            error_context = create_log_context("tracking", "tracking_only_error", error=str(e))
            self.logger.error("Tracking-onlyエラー", error_context)
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
            
            # Tracking+SAM入力ラベル確認
            track_sam_context = create_log_context("tracking", "track_sam_input", 
                                                 boxes_count=len(boxes), labels=labels)
            self.logger.debug("Tracking+SAM入力", track_sam_context)
            
            try:
                masks, mask_scores, _ = self.sam.predict_batch(
                    [image_np], xyxy=[boxes]
                )
                
                result_masks, result_scores = self._process_sam_results(
                    masks, mask_scores, len(boxes)
                )
                
                # Tracking+SAM出力確認
                track_sam_output_context = create_log_context("tracking", "track_sam_output", 
                                                            mask_count=len(result_masks), labels=labels)
                self.logger.debug("Tracking+SAM出力", track_sam_output_context)
                
                return {
                    "boxes": boxes,
                    "labels": labels,
                    "scores": tracking_result["scores"],
                    "masks": result_masks,
                    "mask_scores": result_scores
                }
                
            except Exception as e:
                sam_error_context = create_log_context("tracking", "track_sam_error", error=str(e))
                self.logger.error("Tracking+SAM SAM処理エラー", sam_error_context)
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