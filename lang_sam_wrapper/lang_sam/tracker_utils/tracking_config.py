"""トラッキング設定管理（後方互換性維持・新システム統合）

技術的目的：
- 既存コードとの後方互換性保持
- 新しい統一設定管理システムとの連携
- 段階的リファクタリング支援
"""

import warnings
from typing import Dict, Any, Optional

import logging  # 標準Pythonロギングを使用 - logging_manager.py削除


class TrackingConfig:
    """レガシートラッキング設定管理（後方互換性維持）
    
    注意: このクラスは段階的に廃止予定です。
    新しいコードでは config_manager.TrackingConfig を使用してください。
    """
    
    def __init__(self, bbox_margin: int = 5, bbox_min_size: int = 20, 
                 tracker_min_size: int = 10):
        # 警告出力（本番環境では無効化）
        warnings.warn(
            "TrackingConfig は廃止予定です。config_manager.TrackingConfig を使用してください。",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.bbox_margin = bbox_margin
        self.bbox_min_size = bbox_min_size
        self.tracker_min_size = tracker_min_size
        
        self._logger = logging.getLogger("legacy_tracking_config")  # 標準Pythonロギング
        self._logger.warning("Legacy TrackingConfig is deprecated, consider migrating to ConfigManager")
    
    def update(self, **kwargs) -> None:
        """設定更新（レガシー形式）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self._logger.debug(f"Updated legacy config: {key} = {value}")
    
    # ConfigManager削除に伴い、変換メソッドも削除


# ConfigurationMigrationHelperクラス削除 - ConfigManager削除に伴い不要