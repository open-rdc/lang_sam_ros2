"""カスタム例外クラス - エラーハンドリング統一化"""

from typing import Optional, Any


class LangSAMError(Exception):
    """LangSAM基底例外クラス"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error
        self.message = message
    
    def __str__(self) -> str:
        error_msg = self.message
        if self.error_code:
            error_msg = f"[{self.error_code}] {error_msg}"
        if self.original_error:
            error_msg += f" (Original: {str(self.original_error)})"
        return error_msg


class ModelError(LangSAMError):
    """AIモデル関連エラー"""
    pass


class TrackingError(LangSAMError):
    """トラッキング関連エラー"""
    pass


class ImageProcessingError(LangSAMError):
    """画像処理関連エラー"""
    pass


class ConfigurationError(LangSAMError):
    """設定・パラメータ関連エラー"""
    pass


class ROSError(LangSAMError):
    """ROS通信関連エラー"""
    pass


# 具体的な例外クラス
class SAM2InitError(ModelError):
    """SAM2初期化エラー"""
    
    def __init__(self, model_type: str, checkpoint_path: Optional[str] = None, 
                 original_error: Optional[Exception] = None):
        message = f"SAM2モデル初期化に失敗: {model_type}"
        if checkpoint_path:
            message += f" (checkpoint: {checkpoint_path})"
        super().__init__(message, "SAM2_INIT", original_error)


class GroundingDINOError(ModelError):
    """GroundingDINO関連エラー"""
    
    def __init__(self, operation: str, original_error: Optional[Exception] = None):
        message = f"GroundingDINO処理エラー: {operation}"
        super().__init__(message, "GDINO_ERROR", original_error)


class CSRTTrackingError(TrackingError):
    """CSRTトラッキングエラー"""
    
    def __init__(self, tracker_id: str, operation: str, 
                 original_error: Optional[Exception] = None):
        message = f"CSRTトラッキングエラー [{tracker_id}]: {operation}"
        super().__init__(message, "CSRT_ERROR", original_error)


class ImageConversionError(ImageProcessingError):
    """画像変換エラー"""
    
    def __init__(self, conversion_type: str, shape: Optional[tuple] = None,
                 original_error: Optional[Exception] = None):
        message = f"画像変換エラー: {conversion_type}"
        if shape:
            message += f" (shape: {shape})"
        super().__init__(message, "IMAGE_CONVERT", original_error)


class ParameterValidationError(ConfigurationError):
    """パラメータ検証エラー"""
    
    def __init__(self, param_name: str, expected_type: str, actual_value: Any):
        message = f"パラメータ検証エラー: {param_name} (期待: {expected_type}, 実際: {type(actual_value).__name__})"
        super().__init__(message, "PARAM_INVALID")


# エラーハンドラーユーティリティ
class ErrorHandler:
    """統一エラーハンドリング"""
    
    @staticmethod
    def handle_model_error(operation: str, error: Exception) -> LangSAMError:
        """モデル関連エラーの統一ハンドリング"""
        if "SAM2" in str(error) or "sam2" in str(error):
            return ModelError(f"SAM2エラー: {operation}", "SAM2_ERROR", error)
        elif "grounding" in str(error).lower() or "dino" in str(error).lower():
            return GroundingDINOError(operation, error)
        else:
            return ModelError(f"モデルエラー: {operation}", "MODEL_ERROR", error)
    
    @staticmethod
    def handle_tracking_error(tracker_id: str, operation: str, error: Exception) -> TrackingError:
        """トラッキングエラーの統一ハンドリング"""
        if "CSRT" in str(error) or "tracker" in str(error).lower():
            return CSRTTrackingError(tracker_id, operation, error)
        else:
            return TrackingError(f"トラッキングエラー [{tracker_id}]: {operation}", "TRACK_ERROR", error)
    
    @staticmethod
    def handle_image_error(operation: str, error: Exception, image_shape: Optional[tuple] = None) -> ImageProcessingError:
        """画像処理エラーの統一ハンドリング"""
        if "conversion" in str(error).lower() or "convert" in str(error).lower():
            return ImageConversionError(operation, image_shape, error)
        else:
            return ImageProcessingError(f"画像処理エラー: {operation}", "IMAGE_ERROR", error)
    
    @staticmethod
    def safe_execute(operation: str, func, *args, **kwargs):
        """安全な関数実行（例外ラップ付き）"""
        try:
            return func(*args, **kwargs)
        except LangSAMError:
            # 既にカスタム例外の場合は再発生
            raise
        except Exception as e:
            # 一般的な例外をカスタム例外にラップ
            raise LangSAMError(f"処理エラー: {operation}", "GENERAL_ERROR", e)