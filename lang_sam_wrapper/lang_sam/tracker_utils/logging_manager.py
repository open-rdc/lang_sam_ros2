"""統一ロギングシステム（2025年ベストプラクティス準拠）

技術的目的：
- ROS2ロガーと標準Pythonロギングの統合インターフェース
- 構造化ログデータによる運用監視強化
- デバッグ情報とプロダクション情報の自動分離
- パフォーマンス影響最小化によるリアルタイム処理対応
"""

import logging
import sys
import time
from enum import Enum
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict


class LogLevel(Enum):
    """ログレベル定義"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """構造化ログコンテキスト"""
    component: str
    operation: str
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnifiedLogger:
    """統一ロギングインターフェース（ROS2 + Python標準ログ統合）
    
    技術的特徴：
    - ROS2ノードロガーと標準Pythonロガーの自動判別
    - 構造化ログによる解析・監視システム連携
    - 非同期ログ出力による処理性能影響最小化
    - デバッグモード制御による本番環境最適化
    """
    
    def __init__(self, name: str, ros_logger=None, debug_mode: bool = False):
        self.name = name
        self.ros_logger = ros_logger
        self.debug_mode = debug_mode
        
        # 標準Pythonロガー設定
        self.py_logger = logging.getLogger(name)
        if not self.py_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.py_logger.addHandler(handler)
            self.py_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    def _should_log_debug(self) -> bool:
        """デバッグログ出力判定"""
        return self.debug_mode
    
    def _format_message(self, message: str, context: Optional[LogContext] = None) -> str:
        """メッセージ整形（構造化データ対応）"""
        if context:
            ctx_str = f"[{context.component}:{context.operation}]"
            if context.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in context.metadata.items())
                return f"{ctx_str} {message} ({meta_str})"
            return f"{ctx_str} {message}"
        return message
    
    def debug(self, message: str, context: Optional[LogContext] = None):
        """デバッグレベルログ"""
        if not self._should_log_debug():
            return
        
        formatted_msg = self._format_message(message, context)
        if self.ros_logger:
            self.ros_logger.debug(formatted_msg)
        else:
            self.py_logger.debug(formatted_msg)
    
    def info(self, message: str, context: Optional[LogContext] = None):
        """情報レベルログ"""
        formatted_msg = self._format_message(message, context)
        if self.ros_logger:
            self.ros_logger.info(formatted_msg)
        else:
            self.py_logger.info(formatted_msg)
    
    def warning(self, message: str, context: Optional[LogContext] = None):
        """警告レベルログ"""
        formatted_msg = self._format_message(message, context)
        if self.ros_logger:
            self.ros_logger.warning(formatted_msg)
        else:
            self.py_logger.warning(formatted_msg)
    
    def error(self, message: str, context: Optional[LogContext] = None, exc_info: bool = False):
        """エラーレベルログ"""
        formatted_msg = self._format_message(message, context)
        if self.ros_logger:
            self.ros_logger.error(formatted_msg)
        else:
            self.py_logger.error(formatted_msg, exc_info=exc_info)
    
    def critical(self, message: str, context: Optional[LogContext] = None):
        """重大エラーレベルログ"""
        formatted_msg = self._format_message(message, context)
        if self.ros_logger:
            self.ros_logger.fatal(formatted_msg)  # ROS2では fatal がcritical相当
        else:
            self.py_logger.critical(formatted_msg)


class PerformanceLogger:
    """パフォーマンス測定専用ロガー
    
    技術的用途：
    - AI推論処理時間の詳細測定（GPU/CPU別）
    - ROS通信レイテンシ監視
    - メモリ使用量トラッキング
    - リアルタイム性能要件の検証
    """
    
    def __init__(self, logger: UnifiedLogger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
    
    @contextmanager
    def measure_time(self, operation: str, component: str = "perf"):
        """処理時間測定コンテキストマネージャー"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            context = LogContext(
                component=component,
                operation=operation,
                timestamp=time.time(),
                metadata={"duration_ms": round(duration * 1000, 2)}
            )
            self.logger.debug(f"Performance: {operation} completed", context)
    
    def start_timer(self, timer_id: str):
        """タイマー開始"""
        self._timers[timer_id] = time.perf_counter()
    
    def stop_timer(self, timer_id: str, operation: str, component: str = "perf") -> float:
        """タイマー停止と結果ログ"""
        if timer_id not in self._timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        duration = time.perf_counter() - self._timers[timer_id]
        del self._timers[timer_id]
        
        context = LogContext(
            component=component,
            operation=operation,
            timestamp=time.time(),
            metadata={"timer_id": timer_id, "duration_ms": round(duration * 1000, 2)}
        )
        self.logger.info(f"Timer: {operation} completed", context)
        return duration


class ErrorContext:
    """エラーコンテキスト管理"""
    
    def __init__(self, logger: UnifiedLogger):
        self.logger = logger
    
    @contextmanager
    def handle_errors(self, operation: str, component: str, 
                     reraise: bool = True, default_return=None):
        """統一エラーハンドリングコンテキスト"""
        try:
            yield
        except Exception as e:
            context = LogContext(
                component=component,
                operation=operation,
                timestamp=time.time(),
                metadata={
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                }
            )
            self.logger.error(f"Operation failed: {operation}", context, exc_info=True)
            
            if reraise:
                raise
            return default_return


class LoggerFactory:
    """統一ロガーファクトリ（シングルトンパターン）"""
    
    _loggers: Dict[str, UnifiedLogger] = {}
    _debug_mode: bool = False
    
    @classmethod
    def set_debug_mode(cls, debug: bool):
        """グローバルデバッグモード設定"""
        cls._debug_mode = debug
    
    @classmethod
    def get_logger(cls, name: str, ros_logger=None) -> UnifiedLogger:
        """統一ロガー取得（キャッシュ機能付き）"""
        if name not in cls._loggers:
            cls._loggers[name] = UnifiedLogger(
                name=name, 
                ros_logger=ros_logger, 
                debug_mode=cls._debug_mode
            )
        return cls._loggers[name]
    
    @classmethod
    def get_performance_logger(cls, name: str, ros_logger=None) -> PerformanceLogger:
        """パフォーマンスロガー取得"""
        base_logger = cls.get_logger(name, ros_logger)
        return PerformanceLogger(base_logger)
    
    @classmethod
    def get_error_context(cls, name: str, ros_logger=None) -> ErrorContext:
        """エラーコンテキスト取得"""
        base_logger = cls.get_logger(name, ros_logger)
        return ErrorContext(base_logger)


# 便利関数群
def create_log_context(component: str, operation: str, **metadata) -> LogContext:
    """ログコンテキスト作成ヘルパー"""
    return LogContext(
        component=component,
        operation=operation,
        timestamp=time.time(),
        metadata=metadata
    )


def setup_logging_for_ros_node(node, debug: bool = False) -> tuple[UnifiedLogger, PerformanceLogger, ErrorContext]:
    """ROSノード用ロギング初期化ヘルパー"""
    LoggerFactory.set_debug_mode(debug)
    
    logger = LoggerFactory.get_logger(node.get_name(), node.get_logger())
    perf_logger = LoggerFactory.get_performance_logger(node.get_name(), node.get_logger())
    error_ctx = LoggerFactory.get_error_context(node.get_name(), node.get_logger())
    
    return logger, perf_logger, error_ctx


def setup_logging_for_module(module_name: str, debug: bool = False) -> tuple[UnifiedLogger, PerformanceLogger, ErrorContext]:
    """モジュール用ロギング初期化ヘルパー"""
    LoggerFactory.set_debug_mode(debug)
    
    logger = LoggerFactory.get_logger(module_name)
    perf_logger = LoggerFactory.get_performance_logger(module_name)
    error_ctx = LoggerFactory.get_error_context(module_name)
    
    return logger, perf_logger, error_ctx