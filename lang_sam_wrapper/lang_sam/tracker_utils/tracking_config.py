"""トラッキング設定管理（後方互換性維持・新システム統合）

技術的目的：
- 既存コードとの後方互換性保持
- 新しい統一設定管理システムとの連携
- 段階的リファクタリング支援
"""

import warnings
from typing import Dict, Any, Optional

from .config_manager import ConfigManager, TrackingConfig as NewTrackingConfig
from .logging_manager import LoggerFactory


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
        
        self._logger = LoggerFactory.get_logger("legacy_tracking_config")
        self._logger.warning("Legacy TrackingConfig is deprecated, consider migrating to ConfigManager")
    
    def update(self, **kwargs) -> None:
        """設定更新（レガシー形式）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self._logger.debug(f"Updated legacy config: {key} = {value}")
    
    def to_new_config(self) -> NewTrackingConfig:
        """新設定形式への変換"""
        return NewTrackingConfig(
            bbox_margin=self.bbox_margin,
            bbox_min_size=self.bbox_min_size,
            tracker_min_size=self.tracker_min_size
        )
    
    @classmethod
    def from_new_config(cls, new_config: NewTrackingConfig) -> 'TrackingConfig':
        """新設定形式からの変換"""
        return cls(
            bbox_margin=new_config.bbox_margin,
            bbox_min_size=new_config.bbox_min_size,
            tracker_min_size=new_config.tracker_min_size
        )


class ConfigurationMigrationHelper:
    """設定移行支援ヘルパー"""
    
    @staticmethod
    def migrate_legacy_config(legacy_config: TrackingConfig) -> NewTrackingConfig:
        """レガシー設定の新形式移行"""
        return legacy_config.to_new_config()
    
    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> NewTrackingConfig:
        """辞書からの新設定作成"""
        return NewTrackingConfig(
            bbox_margin=config_dict.get('bbox_margin', 5),
            bbox_min_size=config_dict.get('bbox_min_size', 3),
            tracker_min_size=config_dict.get('tracker_min_size', 3),
            enable_csrt_recovery=config_dict.get('enable_csrt_recovery', True),
            frame_buffer_duration=config_dict.get('frame_buffer_duration', 5.0),
            time_travel_seconds=config_dict.get('time_travel_seconds', 1.0),
            fast_forward_frames=config_dict.get('fast_forward_frames', 10),
            recovery_attempt_frames=config_dict.get('recovery_attempt_frames', 1)
        )
    
    @staticmethod
    def get_unified_config() -> Optional[NewTrackingConfig]:
        """統一設定管理からトラッキング設定取得"""
        config_manager = ConfigManager.get_instance()
        system_config = config_manager.get_config()
        
        if system_config:
            return system_config.tracking
        return None
    
    @staticmethod
    def apply_runtime_updates(updates: Dict[str, Any]) -> bool:
        """実行時設定更新"""
        config_manager = ConfigManager.get_instance()
        return config_manager.update_runtime_config("tracking", updates)