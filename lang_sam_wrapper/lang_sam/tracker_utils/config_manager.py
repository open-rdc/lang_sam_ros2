"""統一設定管理システム（2025年ベストプラクティス準拠）

技術的目的：
- 型安全な設定値バリデーションとランタイム検証
- 環境別設定（開発・本番・テスト）の自動切り替え
- 設定変更の影響範囲可視化と安全な動的更新
- パフォーマンス最適化のための設定プロファイル管理
"""

import os
import yaml
import copy
from typing import Dict, Any, Optional, List, Union, Type, get_type_hints
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path

from .exceptions import ConfigurationError, ErrorSeverity
from .logging_manager import LoggerFactory


class Environment(Enum):
    """動作環境分類"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEBUG = "debug"


class ModelProfile(Enum):
    """AIモデル性能プロファイル"""
    SPEED_OPTIMIZED = "speed"      # 速度重視（リアルタイム用）
    BALANCED = "balanced"          # バランス型（標準）
    ACCURACY_OPTIMIZED = "accuracy" # 精度重視（後処理用）


@dataclass
class ModelConfig:
    """AIモデル設定"""
    sam_model: str = "sam2.1_hiera_tiny"
    text_prompt: str = "white line. red pylon. human. car."
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    tracking_targets: List[str] = field(default_factory=lambda: ["white line", "red pylon", "human", "car"])
    
    def validate(self) -> None:
        """設定値検証"""
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ConfigurationError("box_threshold", self.box_threshold, "must be between 0.0 and 1.0")
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ConfigurationError("text_threshold", self.text_threshold, "must be between 0.0 and 1.0")
        if not self.tracking_targets:
            raise ConfigurationError("tracking_targets", self.tracking_targets, "cannot be empty")
        
        valid_sam_models = ["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base", "sam2.1_hiera_large"]
        if self.sam_model not in valid_sam_models:
            raise ConfigurationError("sam_model", self.sam_model, f"must be one of {valid_sam_models}")


@dataclass
class ExecutionConfig:
    """実行制御設定"""
    gdino_interval_seconds: float = 1.0
    enable_gpu_acceleration: bool = True
    max_concurrent_trackers: int = 10
    processing_timeout_seconds: float = 30.0
    
    def validate(self) -> None:
        """設定値検証"""
        if self.gdino_interval_seconds <= 0:
            raise ConfigurationError("gdino_interval_seconds", self.gdino_interval_seconds, "must be positive")
        if self.max_concurrent_trackers <= 0:
            raise ConfigurationError("max_concurrent_trackers", self.max_concurrent_trackers, "must be positive")
        if self.processing_timeout_seconds <= 0:
            raise ConfigurationError("processing_timeout_seconds", self.processing_timeout_seconds, "must be positive")


@dataclass
class ROSConfig:
    """ROS2通信設定"""
    input_topic: str = "/zed/zed_node/rgb/image_rect_color"
    gdino_topic: str = "/image_gdino"
    csrt_output_topic: str = "/image_csrt"
    sam_topic: str = "/image_sam"
    queue_size: int = 10
    
    def validate(self) -> None:
        """設定値検証"""
        topics = [self.input_topic, self.gdino_topic, self.csrt_output_topic, self.sam_topic]
        for topic in topics:
            if not topic.startswith('/'):
                raise ConfigurationError("topic_name", topic, "must start with '/'")
        if self.queue_size <= 0:
            raise ConfigurationError("queue_size", self.queue_size, "must be positive")


@dataclass
class TrackingConfig:
    """トラッキング制御設定"""
    bbox_margin: int = 5
    bbox_min_size: int = 3
    tracker_min_size: int = 3
    
    def validate(self) -> None:
        """設定値検証"""
        if self.bbox_margin < 0:
            raise ConfigurationError("bbox_margin", self.bbox_margin, "must be non-negative")
        if self.bbox_min_size <= 0:
            raise ConfigurationError("bbox_min_size", self.bbox_min_size, "must be positive")
        if self.tracker_min_size <= 0:
            raise ConfigurationError("tracker_min_size", self.tracker_min_size, "must be positive")


@dataclass
class CSRTConfig:
    """CSRT詳細パラメータ設定（型安全・バリデーション強化）"""
    # 基本特徴量設定
    use_hog: bool = True
    use_color_names: bool = False
    use_gray: bool = True
    use_rgb: bool = False
    use_channel_weights: bool = False
    use_segmentation: bool = True
    
    # 窓関数設定
    window_function: str = "hann"
    kaiser_alpha: float = 3.75
    cheb_attenuation: float = 45.0
    
    # テンプレート設定
    template_size: float = 200.0
    gsl_sigma: float = 1.0
    padding: float = 3.0
    
    # HOG設定
    hog_orientations: int = 9
    hog_clip: float = 0.2
    num_hog_channels_used: int = -1
    
    # 学習率設定
    filter_lr: float = 0.02
    weights_lr: float = 0.02
    
    # ADMM最適化設定
    admm_iterations: int = 4
    
    # ヒストグラム設定
    histogram_bins: int = 16
    histogram_lr: float = 0.04
    background_ratio: int = 2
    
    # スケール設定
    number_of_scales: int = 33
    scale_sigma_factor: float = 0.25
    scale_model_max_area: float = 512.0
    scale_lr: float = 0.025
    scale_step: float = 1.02
    
    # 信頼性設定
    psr_threshold: float = 0.035
    
    def validate(self) -> None:
        """CSRT詳細パラメータ検証"""
        # 窓関数検証
        valid_window_functions = ["hann", "cosine", "uniform", "kaiser", "cheb"]
        if self.window_function not in valid_window_functions:
            raise ConfigurationError("window_function", self.window_function, f"must be one of {valid_window_functions}")
        
        # 数値範囲検証
        if not 0.1 <= self.kaiser_alpha <= 10.0:
            raise ConfigurationError("kaiser_alpha", self.kaiser_alpha, "must be between 0.1 and 10.0")
        if not 10.0 <= self.cheb_attenuation <= 100.0:
            raise ConfigurationError("cheb_attenuation", self.cheb_attenuation, "must be between 10.0 and 100.0")
        if not 50.0 <= self.template_size <= 1000.0:
            raise ConfigurationError("template_size", self.template_size, "must be between 50.0 and 1000.0")
        if not 0.1 <= self.gsl_sigma <= 5.0:
            raise ConfigurationError("gsl_sigma", self.gsl_sigma, "must be between 0.1 and 5.0")
        if not 1.0 <= self.padding <= 10.0:
            raise ConfigurationError("padding", self.padding, "must be between 1.0 and 10.0")
        
        # HOG設定検証
        if not 4 <= self.hog_orientations <= 18:
            raise ConfigurationError("hog_orientations", self.hog_orientations, "must be between 4 and 18")
        if not 0.01 <= self.hog_clip <= 1.0:
            raise ConfigurationError("hog_clip", self.hog_clip, "must be between 0.01 and 1.0")
        
        # 学習率検証
        if not 0.001 <= self.filter_lr <= 0.1:
            raise ConfigurationError("filter_lr", self.filter_lr, "must be between 0.001 and 0.1")
        if not 0.001 <= self.weights_lr <= 0.1:
            raise ConfigurationError("weights_lr", self.weights_lr, "must be between 0.001 and 0.1")
        
        # 反復数検証
        if not 1 <= self.admm_iterations <= 20:
            raise ConfigurationError("admm_iterations", self.admm_iterations, "must be between 1 and 20")
        
        # ヒストグラム設定検証
        if not 8 <= self.histogram_bins <= 64:
            raise ConfigurationError("histogram_bins", self.histogram_bins, "must be between 8 and 64")
        if not 0.001 <= self.histogram_lr <= 0.2:
            raise ConfigurationError("histogram_lr", self.histogram_lr, "must be between 0.001 and 0.2")
        if not 1 <= self.background_ratio <= 10:
            raise ConfigurationError("background_ratio", self.background_ratio, "must be between 1 and 10")
        
        # スケール設定検証
        if not 10 <= self.number_of_scales <= 100:
            raise ConfigurationError("number_of_scales", self.number_of_scales, "must be between 10 and 100")
        if not 0.1 <= self.scale_sigma_factor <= 2.0:
            raise ConfigurationError("scale_sigma_factor", self.scale_sigma_factor, "must be between 0.1 and 2.0")
        if not 128.0 <= self.scale_model_max_area <= 4096.0:
            raise ConfigurationError("scale_model_max_area", self.scale_model_max_area, "must be between 128.0 and 4096.0")
        if not 0.01 <= self.scale_lr <= 0.1:
            raise ConfigurationError("scale_lr", self.scale_lr, "must be between 0.01 and 0.1")
        if not 1.01 <= self.scale_step <= 2.0:
            raise ConfigurationError("scale_step", self.scale_step, "must be between 1.01 and 2.0")
        
        # PSR閾値検証
        if not 0.01 <= self.psr_threshold <= 0.1:
            raise ConfigurationError("psr_threshold", self.psr_threshold, "must be between 0.01 and 0.1")
    
    def get_performance_profile(self) -> ModelProfile:
        """現在の設定からパフォーマンスプロファイルを推定"""
        speed_indicators = 0
        accuracy_indicators = 0
        
        # 速度重視指標
        if not self.use_color_names: speed_indicators += 1
        if self.template_size <= 150.0: speed_indicators += 1
        if self.number_of_scales <= 25: speed_indicators += 1
        if self.admm_iterations <= 3: speed_indicators += 1
        
        # 精度重視指標
        if self.use_color_names: accuracy_indicators += 1
        if self.template_size >= 250.0: accuracy_indicators += 1
        if self.number_of_scales >= 40: accuracy_indicators += 1
        if self.admm_iterations >= 6: accuracy_indicators += 1
        
        if speed_indicators > accuracy_indicators:
            return ModelProfile.SPEED_OPTIMIZED
        elif accuracy_indicators > speed_indicators:
            return ModelProfile.ACCURACY_OPTIMIZED
        else:
            return ModelProfile.BALANCED


@dataclass
class LangSAMSystemConfig:
    """統合システム設定（全コンポーネント統合）"""
    environment: Environment = Environment.DEVELOPMENT
    model: ModelConfig = field(default_factory=ModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    ros: ROSConfig = field(default_factory=ROSConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    csrt: CSRTConfig = field(default_factory=CSRTConfig)
    
    def validate_all(self) -> None:
        """全設定の包括的検証"""
        self.model.validate()
        self.execution.validate()
        self.ros.validate()
        self.tracking.validate()
        self.csrt.validate()
        
        # 相互関係検証
        if self.execution.gdino_interval_seconds < 0.1 and self.csrt.template_size > 300:
            raise ConfigurationError(
                "configuration_mismatch", 
                f"gdino_interval={self.execution.gdino_interval_seconds}, template_size={self.csrt.template_size}",
                "High frequency execution with large template size may cause performance issues"
            )


class ConfigManager:
    """統一設定管理マネージャー（シングルトンパターン）
    
    技術的特徴：
    - YAML設定ファイルからの型安全な自動読み込み
    - 環境変数オーバーライド対応
    - 設定変更の影響分析とロールバック機能
    - パフォーマンスプロファイルの自動推奨
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[LangSAMSystemConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = LoggerFactory.get_logger("config_manager")
            self.initialized = True
            self._config_history: List[LangSAMSystemConfig] = []
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """シングルトンインスタンス取得"""
        return cls()
    
    def load_from_yaml(self, config_path: Union[str, Path], 
                      node_name: str = "lang_sam_tracker_node") -> LangSAMSystemConfig:
        """YAML設定ファイル読み込み（型安全・バリデーション付き）"""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ConfigurationError("config_file", str(config_path), "file not found")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if node_name not in yaml_data or 'ros__parameters' not in yaml_data[node_name]:
                raise ConfigurationError("yaml_structure", node_name, "node configuration not found")
            
            params = yaml_data[node_name]['ros__parameters']
            
            # 環境判定（環境変数または設定ファイルから）
            env_str = os.getenv('LANGSAM_ENV', params.get('environment', 'development'))
            try:
                environment = Environment(env_str)
            except ValueError:
                environment = Environment.DEVELOPMENT
                self.logger.warning(f"Invalid environment '{env_str}', using development")
            
            config = LangSAMSystemConfig(environment=environment)
            
            # 各セクション設定
            self._load_model_config(config.model, params)
            self._load_execution_config(config.execution, params)
            self._load_ros_config(config.ros, params)
            self._load_tracking_config(config.tracking, params)
            self._load_csrt_config(config.csrt, params)
            
            # 全体検証
            config.validate_all()
            
            # 設定履歴保存
            if self._config:
                self._config_history.append(copy.deepcopy(self._config))
            
            self._config = config
            
            self.logger.info(f"Configuration loaded successfully from {config_path}")
            self._log_configuration_summary(config)
            
            return config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            else:
                raise ConfigurationError("config_loading", str(config_path), str(e), e)
    
    def _load_model_config(self, model_config: ModelConfig, params: Dict[str, Any]):
        """モデル設定読み込み"""
        model_config.sam_model = params.get('sam_model', model_config.sam_model)
        model_config.text_prompt = params.get('text_prompt', model_config.text_prompt)
        model_config.box_threshold = float(params.get('box_threshold', model_config.box_threshold))
        model_config.text_threshold = float(params.get('text_threshold', model_config.text_threshold))
        model_config.tracking_targets = params.get('tracking_targets', model_config.tracking_targets)
    
    def _load_execution_config(self, exec_config: ExecutionConfig, params: Dict[str, Any]):
        """実行設定読み込み"""
        exec_config.gdino_interval_seconds = float(params.get('gdino_interval_seconds', exec_config.gdino_interval_seconds))
        exec_config.enable_gpu_acceleration = params.get('enable_gpu_acceleration', exec_config.enable_gpu_acceleration)
        exec_config.max_concurrent_trackers = int(params.get('max_concurrent_trackers', exec_config.max_concurrent_trackers))
        exec_config.processing_timeout_seconds = float(params.get('processing_timeout_seconds', exec_config.processing_timeout_seconds))
    
    def _load_ros_config(self, ros_config: ROSConfig, params: Dict[str, Any]):
        """ROS設定読み込み"""
        ros_config.input_topic = params.get('input_topic', ros_config.input_topic)
        ros_config.gdino_topic = params.get('gdino_topic', ros_config.gdino_topic)
        ros_config.csrt_output_topic = params.get('csrt_output_topic', ros_config.csrt_output_topic)
        ros_config.sam_topic = params.get('sam_topic', ros_config.sam_topic)
        ros_config.queue_size = int(params.get('queue_size', ros_config.queue_size))
    
    def _load_tracking_config(self, track_config: TrackingConfig, params: Dict[str, Any]):
        """トラッキング設定読み込み"""
        track_config.bbox_margin = int(params.get('bbox_margin', track_config.bbox_margin))
        track_config.bbox_min_size = int(params.get('bbox_min_size', track_config.bbox_min_size))
        track_config.tracker_min_size = int(params.get('tracker_min_size', track_config.tracker_min_size))
    
    def _load_csrt_config(self, csrt_config: CSRTConfig, params: Dict[str, Any]):
        """CSRT設定読み込み（プレフィックス除去対応）"""
        csrt_params = {}
        for key, value in params.items():
            if key.startswith('csrt_'):
                clean_key = key[5:]  # 'csrt_'プレフィックス除去
                csrt_params[clean_key] = value
        
        # データクラスフィールドとマッピング
        for field in fields(CSRTConfig):
            if field.name in csrt_params:
                value = csrt_params[field.name]
                # 型変換
                if field.type == bool:
                    value = bool(value)
                elif field.type == int:
                    value = int(value)
                elif field.type == float:
                    value = float(value)
                elif field.type == str:
                    value = str(value)
                
                setattr(csrt_config, field.name, value)
    
    def _log_configuration_summary(self, config: LangSAMSystemConfig):
        """設定概要ログ出力"""
        self.logger.info("=== Configuration Summary ===")
        self.logger.info(f"Environment: {config.environment.value}")
        self.logger.info(f"SAM Model: {config.model.sam_model}")
        self.logger.info(f"GDINO Interval: {config.execution.gdino_interval_seconds}s")
        self.logger.info(f"Tracking Targets: {len(config.model.tracking_targets)} objects")
        
        profile = config.csrt.get_performance_profile()
        self.logger.info(f"CSRT Profile: {profile.value}")
        
        self.logger.info("=== End Configuration Summary ===")
    
    def get_config(self) -> Optional[LangSAMSystemConfig]:
        """現在の設定取得"""
        return self._config
    
    def rollback_config(self) -> bool:
        """設定ロールバック"""
        if not self._config_history:
            self.logger.warning("No configuration history available for rollback")
            return False
        
        previous_config = self._config_history.pop()
        self._config = previous_config
        self.logger.info("Configuration rolled back to previous version")
        return True
    
    def update_runtime_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """実行時設定更新（安全性確認付き）"""
        if not self._config:
            self.logger.error("No configuration loaded")
            return False
        
        try:
            # 更新前の設定をバックアップ
            backup_config = copy.deepcopy(self._config)
            
            # セクション別更新
            if section == "model":
                for key, value in updates.items():
                    if hasattr(self._config.model, key):
                        setattr(self._config.model, key, value)
                self._config.model.validate()
            elif section == "execution":
                for key, value in updates.items():
                    if hasattr(self._config.execution, key):
                        setattr(self._config.execution, key, value)
                self._config.execution.validate()
            elif section == "csrt":
                for key, value in updates.items():
                    if hasattr(self._config.csrt, key):
                        setattr(self._config.csrt, key, value)
                self._config.csrt.validate()
            else:
                raise ConfigurationError("invalid_section", section, "unknown configuration section")
            
            # 全体整合性確認
            self._config.validate_all()
            
            self.logger.info(f"Runtime configuration updated: {section}")
            return True
            
        except Exception as e:
            # エラー時はロールバック
            self._config = backup_config
            self.logger.error(f"Configuration update failed, rolled back: {e}")
            return False
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """パフォーマンス最適化推奨設定"""
        if not self._config:
            return {}
        
        current_profile = self._config.csrt.get_performance_profile()
        recommendations = {
            "current_profile": current_profile.value,
            "suggestions": []
        }
        
        if self._config.environment == Environment.PRODUCTION:
            if current_profile != ModelProfile.SPEED_OPTIMIZED:
                recommendations["suggestions"].append({
                    "type": "performance",
                    "description": "本番環境では速度最適化を推奨",
                    "settings": {
                        "csrt.use_color_names": False,
                        "csrt.template_size": 150.0,
                        "csrt.number_of_scales": 25,
                        "csrt.admm_iterations": 3
                    }
                })
        
        if self._config.execution.gdino_interval_seconds < 0.5:
            recommendations["suggestions"].append({
                "type": "stability",
                "description": "高頻度実行時はテンプレートサイズ縮小を推奨",
                "settings": {
                    "csrt.template_size": 150.0
                }
            })
        
        return recommendations


# 便利関数群
def load_default_config() -> LangSAMSystemConfig:
    """デフォルト設定読み込み"""
    return LangSAMSystemConfig()


def create_speed_optimized_config() -> LangSAMSystemConfig:
    """速度最適化設定生成"""
    config = LangSAMSystemConfig()
    config.csrt.use_color_names = False
    config.csrt.template_size = 150.0
    config.csrt.number_of_scales = 25
    config.csrt.admm_iterations = 3
    return config


def create_accuracy_optimized_config() -> LangSAMSystemConfig:
    """精度最適化設定生成"""
    config = LangSAMSystemConfig()
    config.csrt.use_color_names = True
    config.csrt.use_segmentation = True
    config.csrt.template_size = 300.0
    config.csrt.number_of_scales = 50
    config.csrt.admm_iterations = 6
    return config