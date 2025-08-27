import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from lang_sam.models.utils import DEVICE

# GroundingDINO: テキストプロンプトベースのゼロショット物体検出モデル
# Transformerアーキテクチャを使用し、自然言語記述から物体を検出する目的で使用

class GDINO:
    """GroundingDINOラッパークラス
    
    HuggingFace Transformersライブラリを使用してGroundingDINOモデルを
    管理・実行する目的で使用。事前学習済みモデルの自動ダウンロード機能を提供。
    """
    
    def build_model(self, ckpt_path: str | None = None, device=DEVICE):
        """モデルとプロセッサーの初期化
        
        Args:
            ckpt_path: カスタムチェックポイントパス（Noneの場合はHuggingFaceから自動取得）
            device: 推論実行デバイス（CUDA/CPU自動選択）
        
        技術的実装：
        - AutoProcessorで画像とテキストの前処理パイプラインを構築
        - AutoModelForZeroShotObjectDetectionでTransformerモデルをロード
        - GPU/CPUデバイスへの自動配置により推論速度を最適化する目的で使用
        """
        # HuggingFace Hub上の事前学習済みモデルを使用（デフォルト）
        # IDEA-Research提供のbase版を使用することで精度と速度のバランスを取る目的で使用
        model_id = "IDEA-Research/grounding-dino-base" if ckpt_path is None else ckpt_path
        
        # プロセッサー初期化：画像正規化とテキストトークン化を統合管理する目的で使用
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # モデル初期化：Transformerベースの検出モデルをGPUメモリにロードする目的で使用
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            device
        )

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        """ゼロショット物体検出の実行
        
        Args:
            images_pil: PIL形式の入力画像リスト
            texts_prompt: 検出対象を記述したテキストプロンプト（例："white line. red pylon."）
            box_threshold: バウンディングボックスの信頼度閾値（低いほど多くの検出）
            text_threshold: テキスト類似度の閾値（低いほど緩い一致）
        
        Returns:
            検出結果のリスト（boxes, labels, scores含む）
            
        技術的処理：
        1. プロンプトの末尾にピリオド追加（モデルの学習データ形式に合わせる目的）
        2. 画像とテキストをテンソル化してGPUへ転送
        3. 推論実行（勾配計算無効化で高速化）
        4. 後処理で座標変換と閾値フィルタリング
        """
        # プロンプトの形式統一：末尾ピリオドを確保
        # GroundingDINOの学習時データ形式に合わせることで検出精度を向上させる目的で使用
        for i, prompt in enumerate(texts_prompt):
            if prompt[-1] != ".":
                texts_prompt[i] += "."
        # 前処理：画像の正規化とテキストのトークン化を同時実行する目的で使用
        # return_tensors="pt"でPyTorchテンソル形式に変換
        inputs = self.processor(images=images_pil, text=texts_prompt, return_tensors="pt").to(self.model.device)
        
        # 推論実行：勾配計算を無効化してメモリ消費とを削減する目的で使用
        with torch.no_grad():
            outputs = self.model(**inputs)  # Transformerフォワードパス実行

        # 後処理：モデル出力を実用的な検出結果に変換
        # - NMSによる重複除去
        # - 閾値によるフィルタリング
        # - 画像座標系への変換
        # target_sizesで元画像サイズに合わせた座標変換を行う目的で使用
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,    # IoU閾値でバウンディングボックスをフィルタ
            text_threshold=text_threshold,  # コサイン類似度でテキスト一致度をフィルタ
            target_sizes=[k.size[::-1] for k in images_pil],  # (width, height) → (height, width)
        )
        return results


if __name__ == "__main__":
    # デバッグ用コード（本番環境では無効）
    pass
