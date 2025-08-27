"""Language-SAM統合トラッカー：高レベルAPI（アプリケーション層）

技術的設計：
- ModelCoordinatorへの処理委譲によるレイヤー分離
- 簡潔なAPIインターフェースで複雑なAI推論パイプラインを抽象化
- ROS2ノードとの疎結合設計による再利用性確保
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional

from ..models.utils import DEVICE
from .tracking_config import TrackingConfig
from .model_coordinator import ModelCoordinator
# カスタム例外削除 - 標準Exception使用


class LangSAMTracker:
    """Language Segment-Anything統合システム：高レベルAPIファサード
    
    技術アーキテクチャ：
    - Facade Pattern: 複雑なAI推論システムの単純化されたインターフェース提供
    - Delegation Pattern: ModelCoordinatorへの処理委譲による責任分離
    - Configuration Management: デフォルト設定とカスタム設定の柔軟な管理
    
    対象ユーザー：
    - ROSノード開発者（簡単な関数呼び出しでAI推論実行）
    - アプリケーション開発者（内部実装を意識せずに利用可能）
    """
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        """統合AIシステム初期化：3つのAIモデルの協調環境構築
        
        技術的初期化処理：
        - SAM2モデル選択（hiera_tiny/small/base/large）とGPU配置
        - GroundingDINOモデルの事前学習重みロード
        - CSRTトラッキングシステムのデフォルト設定
        - メモリ効率とレイテンシのバランス最適化
        """
        self.sam_type = sam_type
        
        self.coordinator = ModelCoordinator(sam_type, ckpt_path, device)
        
        self._default_tracking_targets = ["white line", "red pylon", "human", "car"]
        self._setup_default_tracking()
    
    def _setup_default_tracking(self) -> None:
        """フォーミュラカー環境に最適化されたデフォルトトラッキング設定
        
        技術的パラメータ選択根拠：
        - bbox_margin=5: トラッキング精度とロバスト性のバランス
        - bbox_min_size=20: 小さな物体のノイズ除去
        - tracker_min_size=10: CSRT最小追跡サイズ（計算効率との兼ね合い）
        - 対象物体: フォーミュラカー競技に特化した要素の設定
        """
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
        """フル推論パイプライン実行：3段階AI処理の統合制御
        
        技術的処理フロー：
        1. 自然言語→視覚的検出（GroundingDINO）：マルチモーダルTransformer推論
        2. 検出結果→リアルタイム追跡（CSRT）：相関フィルタベース高速処理
        3. 追跡BBox→精密セグメンテーション（SAM2）：ビジョンTransformerマスク生成
        
        パフォーマンス制御：
        - update_trackers: トラッキング状態更新の有効/無効制御
        - run_sam: 重いセグメンテーション処理のオン/オフ切り替え
        """
        try:
            return self.coordinator.predict_full_pipeline(
                images_pil, texts_prompt, box_threshold, text_threshold,
                update_trackers, run_sam
            )
        except Exception as e:
            raise RuntimeError(f"LangSAMトラッカー推論失敗: {str(e)}")
    
    def update_trackers_only(self, image_np: np.ndarray) -> Dict:
        """軽量CSRTトラッキング：30Hzリアルタイム処理特化
        
        技術的最適化：
        - GPU推論を除外したCPUベースの高速処理
        - HOG+色ヒストグラム特徴量による軽量マッチング
        - メモリ使用量最小化（SAM2/GroundingDINOのスキップ）
        - フレーム間短時間遅延での続続追跡実現
        """
        try:
            return self.coordinator.update_tracking_only(image_np)
        except Exception as e:
            raise RuntimeError(f"トラッキング専用モード失敗: {str(e)}")
    
    def update_trackers_with_sam(self, image_np: np.ndarray) -> Dict:
        """ハイブリッド処理：CSRT追跡+SAM2セグメンテーションの最適組み合わせ
        
        技術的アーキテクチャ：
        1. 軽量CSRTアルゴリズムで物体位置追跡（CPU高速処理）
        2. 追跡BBoxをSAM2プロンプトとして入力（精密な空間定位）
        3. Vision Transformerによる高精度マスク生成（GPU並列処理）
        
        使用目的：
        - 精密な物体輪郭抜き出しが必要なタスク
        - トラッキング精度とセグメンテーション品質の両立
        """
        try:
            return self.coordinator.update_tracking_with_sam(image_np)
        except Exception as e:
            raise RuntimeError(f"トラッキング+セグメンテーション失敗: {str(e)}")
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        """動的追跡対象設定：実行時ターゲット変更機能
        
        技術的仕組み：
        - ラベルフィルタリング：GroundingDINO検出結果から指定クラスのみ抽出
        - トラッキングマネージャでの動的フィルタ適用
        - フォーミュラカー環境でのシーン別物体追跡切り替え
        """
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_manager.set_tracking_targets(targets)
    
    def set_tracking_config(self, config: Dict[str, int]) -> None:
        """トラッキングアルゴリズムの動的パラメータ調整
        
        調整可能な技術的パラメータ：
        - bbox_margin: トラッキング精度とロバスト性のバランシング
        - bbox_min_size: 小さな物体のノイズ除去闾値
        - tracker_min_size: CSRTアルゴリズムの最小追跡サイズ
        
        目的：異なるカメラ解像度や物体サイズに対する適応的最適化
        """
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_config.update(**config)
    
    def set_csrt_params(self, csrt_params: Dict) -> None:
        """相関フィルタベーストラッキングの内部パラメータ24項目詳細調整
        
        技術的カスタマイズ対象：
        - HOG特徴量: セルサイズ、ブロックサイズ、ビン数調整
        - 色ヒストグラム: ビン数、空間重み付けパラメータ
        - 相関フィルタ: 学習率、探索範囲、スケール数の微細調整
        
        適用シーン：
        - 特定物体の追跡精度向上（車両、人物、コーンなど）
        - カメラの動きや照明変化に対するロバスト性向上
        """
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
        """全トラッカー状態クリア"""
        self.coordinator.clear_tracking_state()
    
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