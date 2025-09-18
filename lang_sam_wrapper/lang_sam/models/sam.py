import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

from lang_sam.models.utils import DEVICE

# SAM2 (Segment Anything Model 2): Meta AI製の汎用セグメンテーションモデル
# ゼロショットで物体のピクセルレベルマスクを生成する目的で使用

# SAM2モデルバリアント定義
# モデルサイズと精度のトレードオフを選択する目的で使用
SAM_MODELS = {
    "sam2.1_hiera_tiny": {  # 最軽量版：リアルタイム処理向け
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    },
    "sam2.1_hiera_small": {  # 軽量版：速度重視
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_base_plus": {  # バランス版：精度と速度のバランス
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_large": {  # 最高精度版：精度最優先
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}


class SAM:
    """SAM2モデルラッパークラス
    
    Hierarchical Vision Transformerアーキテクチャを使用したSAM2モデルを
    管理・実行する目的で使用。ボックス/ポイントプロンプトからセグメンテーションを生成。
    """
    
    def build_model(self, sam_type: str, ckpt_path: str | None = None, device=DEVICE):
        """モデル初期化とチェックポイント読み込み
        
        Args:
            sam_type: モデルバリアント名 (tiny/small/base_plus/large)
            ckpt_path: カスタムチェックポイントパス（NoneでMeta AIから自動DL）
            device: 推論デバイス (CUDA/CPU)
        
        技術的実装：
        - Hydraで設定管理を行い、モデルアーキテクチャを動的構築する目的で使用
        - SAM2ImagePredictor: バウンディングボックスベースのセグメンテーション
        - SAM2AutomaticMaskGenerator: 全画面自動セグメンテーション
        """
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        
        # Hydra設定ファイルからモデル構成をロード
        # YAML形式の設定を解析してモデルアーキテクチャを定義する目的で使用
        cfg = compose(config_name=SAM_MODELS[self.sam_type]["config"], overrides=[])
        OmegaConf.resolve(cfg)  # 参照解決で設定値を完全展開
        # モデルインスタンス化：設定からVision Transformerモデルを動的構築する目的で使用
        self.model = instantiate(cfg.model, _recursive_=True)
        
        # チェックポイント読み込み：事前学習済み重みをロードする目的で使用
        self._load_checkpoint(self.model)
        
        # GPU転送と評価モード設定：推論高速化の目的で使用
        self.model = self.model.to(device)
        self.model.eval()  # DropoutやBatchNormを推論モードに設定
        
        # セグメンテーションモジュール初期化
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)  # 全画面自動セグメンテーション用
        self.predictor = SAM2ImagePredictor(self.model)              # プロンプトベースセグメンテーション用

    def _load_checkpoint(self, model: torch.nn.Module):
        """チェックポイントからモデル重みをロード
        
        Meta AIの公式チェックポイントまたはカスタムチェックポイントから
        事前学習済み重みを読み込む目的で使用。
        """
        if self.ckpt_path is None:
            # Meta AIの公式サーバーから自動ダウンロードする目的で使用
            checkpoint_url = SAM_MODELS[self.sam_type]["url"]
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]
        else:
            # ローカルファイルからカスタムチェックポイントを読み込む目的で使用
            checkpoint_url = self.ckpt_path
            state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)["model"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                re-downloading it. Error: {e}")

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Output format
        SAM2AutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information
        about the mask:

        segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        area - [int] - the area of the mask in pixels
        bbox - [List[int]] - the boundary box of the mask in xywh format
        predicted_iou - [float] - the model's own prediction for the quality of the mask
        point_coords - [List[List[float]]] - the sampled input point that generated this mask
        stability_score - [float] - an additional measure of mask quality
        crop_box - List[int] - the crop of the image used to generate this mask in xywh format
        """

        sam2_result = self.mask_generator.generate(image_rgb)
        return sam2_result

    def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """バウンディングボックスからセグメンテーションマスクを生成
        
        Args:
            image_rgb: RGB形式の入力画像
            xyxy: バウンディングボックス座標 [x1, y1, x2, y2]
        
        Returns:
            masks: セグメンテーションマスク (H, W)
            scores: マスク品質スコア
            logits: モデル出力のlogit値
        
        技術的処理：
        - set_imageで画像特徴を一度抽出してキャッシュする目的で使用
        - multimask_output=Falseで単一最良マスクのみを出力する目的で使用
        """
        # 画像特徴抽出：ViTエンコーダーで画像特徴を事前計算してキャッシュする目的で使用
        self.predictor.set_image(image_rgb)
        
        # セグメンテーション実行：ボックスプロンプトからマスクを生成する目的で使用
        masks, scores, logits = self.predictor.predict(box=xyxy, multimask_output=False)
        
        # 次元削減：不要な次元を除去して2Dマスクにする目的で使用
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks, scores, logits

    def predict_batch(
        self,
        images_rgb: list[np.ndarray],
        xyxy: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        self.predictor.set_image_batch(images_rgb)

        masks, scores, logits = self.predictor.predict_batch(box_batch=xyxy, multimask_output=False)

        masks = [np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks]
        scores = [np.squeeze(score) for score in scores]
        logits = [np.squeeze(logit, axis=1) if len(logit.shape) > 3 else logit for logit in logits]
        return masks, scores, logits

    def predict_with_points(
        self, 
        image_rgb: np.ndarray, 
        point_coords: np.ndarray, 
        point_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point coordinates入力でのセグメンテーション予測
        
        Args:
            image_rgb: RGB画像 (H, W, 3)
            point_coords: 点座標 [[x1, y1], [x2, y2], ...] shape: (N, 2)
            point_labels: 点ラベル [1, 1, ...] (1=前景, 0=背景) shape: (N,)
            
        Returns:
            masks: セグメンテーションマスク
            scores: 信頼度スコア
            logits: ロジット値
        """
        self.predictor.set_image(image_rgb)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks, scores, logits
