"""LangSAM例外クラス群（2025年ベストプラクティス準拠）

技術的目的：
- 構造化例外情報による詳細なエラートレーサビリティ
- 統一ロギングシステムとの密接な連携
- 運用監視システム向けの機械可読エラーコード体系
- 復旧可能エラーと致命的エラーの明確な分類
"""

import time
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """エラー重要度分類"""
    LOW = "LOW"           # 警告レベル、処理継続可能
    MEDIUM = "MEDIUM"     # 一部機能停止、代替処理可能
    HIGH = "HIGH"         # 重要機能停止、手動復旧必要
    CRITICAL = "CRITICAL" # システム全体停止、即座の対応必要


@dataclass
class ErrorMetadata:
    """構造化エラーメタデータ"""
    timestamp: float
    component: str
    operation: str
    severity: ErrorSeverity
    recoverable: bool
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": self.context
        }


class LangSAMError(Exception):
    """LangSAM基底例外クラス（構造化エラー情報付き）"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 original_error: Optional[Exception] = None,
                 component: str = "unknown",
                 operation: str = "unknown",
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recoverable: bool = True,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error
        self.message = message
        
        # 構造化メタデータ
        self.metadata = ErrorMetadata(
            timestamp=time.time(),
            component=component,
            operation=operation,
            severity=severity,
            recoverable=recoverable,
            context=context or {}
        )
        
        # トレースバック情報保存
        self.error_traceback = traceback.format_exc() if original_error else None
    
    def __str__(self) -> str:
        error_msg = self.message
        if self.error_code:
            error_msg = f"[{self.error_code}] {error_msg}"
        if self.original_error:
            error_msg += f" (Original: {str(self.original_error)})"
        return error_msg
    
    def get_structured_info(self) -> Dict[str, Any]:
        """構造化エラー情報取得"""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "original_error": str(self.original_error) if self.original_error else None,
            "metadata": self.metadata.to_dict(),
            "traceback": self.error_traceback
        }
    
    def is_recoverable(self) -> bool:
        """復旧可能性判定"""
        return self.metadata.recoverable
    
    def get_severity(self) -> ErrorSeverity:
        """エラー重要度取得"""
        return self.metadata.severity


class ModelError(LangSAMError):
    """AIモデル関連エラー"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 original_error: Optional[Exception] = None,
                 operation: str = "model_operation",
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            original_error=original_error,
            component="ai_model",
            operation=operation,
            severity=ErrorSeverity.HIGH,
            recoverable=True,
            context=context
        )


class TrackingError(LangSAMError):
    """トラッキング関連エラー"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 original_error: Optional[Exception] = None,
                 operation: str = "tracking_operation",
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            original_error=original_error,
            component="tracking",
            operation=operation,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            context=context
        )


class ROSCommunicationError(LangSAMError):
    """ROS通信関連エラー"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 original_error: Optional[Exception] = None,
                 operation: str = "ros_operation",
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            original_error=original_error,
            component="ros_communication",
            operation=operation,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            context=context
        )


class SAM2InitError(ModelError):
    """SAM2初期化エラー"""
    
    def __init__(self, model_type: str, checkpoint_path: Optional[str] = None, 
                 original_error: Optional[Exception] = None):
        message = f"SAM2モデル初期化に失敗: {model_type}"
        if checkpoint_path:
            message += f" (checkpoint: {checkpoint_path})"
        
        context = {
            "model_type": model_type,
            "checkpoint_path": checkpoint_path
        }
        
        super().__init__(
            message=message,
            error_code="SAM2_INIT",
            original_error=original_error,
            operation="sam2_initialization",
            context=context
        )


class GroundingDINOError(ModelError):
    """GroundingDINO関連エラー"""
    
    def __init__(self, operation: str, original_error: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None):
        message = f"GroundingDINO処理エラー: {operation}"
        
        super().__init__(
            message=message,
            error_code="GDINO_ERROR",
            original_error=original_error,
            operation=f"gdino_{operation}",
            context=context or {}
        )


class CSRTTrackingError(TrackingError):
    """CSRTトラッキングエラー"""
    
    def __init__(self, tracker_id: str, operation: str, 
                 original_error: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None):
        message = f"CSRTトラッキングエラー [{tracker_id}]: {operation}"
        
        tracking_context = {"tracker_id": tracker_id}
        if context:
            tracking_context.update(context)
        
        super().__init__(
            message=message,
            error_code="CSRT_ERROR",
            original_error=original_error,
            operation=f"csrt_{operation}",
            context=tracking_context
        )


class ConfigurationError(LangSAMError):
    """設定関連エラー"""
    
    def __init__(self, parameter: str, value: Any, reason: str,
                 original_error: Optional[Exception] = None):
        message = f"設定エラー: {parameter} = {value} ({reason})"
        
        context = {
            "parameter": parameter,
            "value": str(value),
            "reason": reason
        }
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            original_error=original_error,
            component="configuration",
            operation="parameter_validation",
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            context=context
        )


class ResourceError(LangSAMError):
    """リソース関連エラー（GPU、メモリ、ファイル等）"""
    
    def __init__(self, resource_type: str, operation: str, reason: str,
                 original_error: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None):
        message = f"リソースエラー [{resource_type}]: {operation} - {reason}"
        
        resource_context = {
            "resource_type": resource_type,
            "reason": reason
        }
        if context:
            resource_context.update(context)
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            original_error=original_error,
            component="resource_management",
            operation=f"resource_{operation}",
            severity=ErrorSeverity.CRITICAL,
            recoverable=True,
            context=resource_context
        )


class ErrorHandler:
    """統一エラーハンドリング（拡張版）"""
    
    @staticmethod
    def handle_model_error(operation: str, error: Exception, 
                          context: Optional[Dict[str, Any]] = None) -> ModelError:
        """モデル関連エラーの統一ハンドリング"""
        if "SAM2" in str(error) or "sam2" in str(error):
            return ModelError(
                f"SAM2エラー: {operation}", 
                "SAM2_ERROR", 
                error, 
                operation, 
                context
            )
        elif "grounding" in str(error).lower() or "dino" in str(error).lower():
            return GroundingDINOError(operation, error, context)
        else:
            return ModelError(
                f"モデルエラー: {operation}", 
                "MODEL_ERROR", 
                error, 
                operation, 
                context
            )
    
    @staticmethod
    def handle_tracking_error(tracker_id: str, operation: str, error: Exception,
                            context: Optional[Dict[str, Any]] = None) -> TrackingError:
        """トラッキングエラーの統一ハンドリング"""
        if "CSRT" in str(error) or "tracker" in str(error).lower():
            return CSRTTrackingError(tracker_id, operation, error, context)
        else:
            return TrackingError(
                f"トラッキングエラー [{tracker_id}]: {operation}", 
                "TRACK_ERROR", 
                error, 
                operation, 
                context
            )
    
    @staticmethod
    def handle_ros_error(operation: str, error: Exception,
                        context: Optional[Dict[str, Any]] = None) -> ROSCommunicationError:
        """ROS通信エラーの統一ハンドリング"""
        return ROSCommunicationError(
            f"ROS通信エラー: {operation}",
            "ROS_ERROR",
            error,
            operation,
            context
        )
    
    @staticmethod
    def safe_execute(operation: str, func, component: str = "general",
                    reraise: bool = True, default_return=None, *args, **kwargs):
        """安全な関数実行（拡張例外ラップ付き）"""
        try:
            return func(*args, **kwargs)
        except LangSAMError:
            if reraise:
                raise
            return default_return
        except Exception as e:
            wrapped_error = LangSAMError(
                f"処理エラー: {operation}",
                "GENERAL_ERROR",
                e,
                component,
                operation,
                ErrorSeverity.MEDIUM,
                True
            )
            if reraise:
                raise wrapped_error
            return default_return
    
    @staticmethod
    def create_error_recovery_plan(error: LangSAMError) -> Dict[str, Any]:
        """エラー復旧計画生成"""
        recovery_plan = {
            "recoverable": error.is_recoverable(),
            "severity": error.get_severity().value,
            "component": error.metadata.component,
            "suggested_actions": []
        }
        
        # コンポーネント別復旧提案
        if error.metadata.component == "ai_model":
            recovery_plan["suggested_actions"] = [
                "モデル再初期化を試行",
                "GPU/CPUメモリクリア",
                "代替モデルへの切り替え検討"
            ]
        elif error.metadata.component == "tracking":
            recovery_plan["suggested_actions"] = [
                "トラッカー再初期化",
                "検出結果からトラッキング再開",
                "パラメータ調整の検討"
            ]
        elif error.metadata.component == "ros_communication":
            recovery_plan["suggested_actions"] = [
                "ノード再起動",
                "トピック接続状態確認",
                "ネットワーク接続確認"
            ]
        
        return recovery_plan