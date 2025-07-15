"""
GPU リソース管理: GPUメモリとプロセス管理の最適化
"""

import torch
import psutil
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import queue
import gc

class GPUPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class GPUTask:
    """GPU処理タスクのラッパー"""
    def __init__(self, task_id: str, priority: GPUPriority, 
                 estimated_memory: int, func: Callable, args: tuple, kwargs: dict):
        self.task_id = task_id
        self.priority = priority
        self.estimated_memory = estimated_memory  # MB
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.created_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result = None
        self.error = None

class GPUResourceManager:
    """GPU リソース管理クラス"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # GPU情報の取得
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # MB
            self.device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU検出: {self.device_name}, 総メモリ: {self.total_memory}MB")
        else:
            self.total_memory = 0
            self.device_name = "CPU"
            self.logger.warning("GPU not available, using CPU")
        
        # リソース管理
        self.memory_threshold = 0.8  # メモリ使用率の閾値
        self.cleanup_threshold = 0.9  # クリーンアップ実行の閾値
        self.monitoring_interval = 1.0  # 監視間隔（秒）
        
        # タスクキューとロック
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.resource_lock = threading.RLock()
        self.active_tasks: Dict[str, GPUTask] = {}
        self.completed_tasks: List[GPUTask] = []
        
        # 優先度管理
        self.priority_weights = {
            GPUPriority.HIGH: 1,
            GPUPriority.MEDIUM: 2,
            GPUPriority.LOW: 3
        }
        
        # 統計情報（監視スレッド開始前に初期化）
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'memory_cleanups': 0,
            'peak_memory_usage': 0
        }
        
        # GPU処理用のエグゼキューター
        self.gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GPUProcessor")
        
        # 監視スレッド
        self.monitoring_thread = threading.Thread(target=self._monitor_gpu_usage, daemon=True)
        self.monitoring_active = True
        self.monitoring_thread.start()
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """現在のGPUメモリ使用量を取得 (使用量MB, 使用率%)"""
        if not self.gpu_available:
            return 0.0, 0.0
        
        try:
            used_memory = torch.cuda.memory_allocated(0) // (1024 * 1024)  # MB
            usage_percent = (used_memory / self.total_memory) * 100
            return used_memory, usage_percent
        except Exception as e:
            self.logger.error(f"GPU メモリ使用量取得エラー: {e}")
            return 0.0, 0.0
    
    def is_memory_available(self, required_memory: int) -> bool:
        """指定されたメモリ量が利用可能かチェック"""
        if not self.gpu_available:
            return True  # CPUモードでは常に許可
        
        used_memory, usage_percent = self.get_memory_usage()
        available_memory = self.total_memory - used_memory
        
        # メモリ閾値とリクエストされたメモリ量をチェック
        return (usage_percent < self.memory_threshold * 100 and 
                available_memory >= required_memory)
    
    def submit_task(self, task_id: str, priority: GPUPriority, 
                   estimated_memory: int, func: Callable, 
                   *args, **kwargs) -> GPUTask:
        """GPU処理タスクを投入"""
        task = GPUTask(task_id, priority, estimated_memory, func, args, kwargs)
        
        with self.resource_lock:
            self.stats['total_tasks'] += 1
            
            # 優先度に基づいてキューに追加
            priority_value = self.priority_weights[priority]
            self.task_queue.put((priority_value, task.created_time, task))
            
            self.logger.info(f"GPU タスク投入: {task_id} (優先度: {priority.value}, 推定メモリ: {estimated_memory}MB)")
        
        return task
    
    def process_tasks(self) -> None:
        """タスクキューを処理"""
        while True:
            try:
                # キューからタスクを取得
                _, _, task = self.task_queue.get(timeout=1.0)
                
                # メモリ使用量をチェック
                if not self.is_memory_available(task.estimated_memory):
                    self.logger.warning(f"メモリ不足によりタスク {task.task_id} を延期")
                    # タスクを再度キューに戻す
                    self.task_queue.put((self.priority_weights[task.priority], task.created_time, task))
                    time.sleep(0.1)  # 短い待機
                    continue
                
                # タスクを実行
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"タスク処理エラー: {e}")
    
    def _execute_task(self, task: GPUTask) -> None:
        """タスクを実行"""
        try:
            with self.resource_lock:
                self.active_tasks[task.task_id] = task
                task.start_time = time.time()
            
            self.logger.debug(f"GPU タスク実行開始: {task.task_id}")
            
            # メモリクリーンアップ（必要に応じて）
            used_memory, usage_percent = self.get_memory_usage()
            if usage_percent > self.cleanup_threshold * 100:
                self._cleanup_gpu_memory()
            
            # 実際のタスク実行
            task.result = task.func(*task.args, **task.kwargs)
            
            task.end_time = time.time()
            execution_time = task.end_time - task.start_time
            
            self.logger.info(f"GPU タスク完了: {task.task_id} (実行時間: {execution_time:.3f}秒)")
            
            with self.resource_lock:
                self.stats['completed_tasks'] += 1
                self.completed_tasks.append(task)
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
            
        except Exception as e:
            task.error = e
            task.end_time = time.time()
            
            self.logger.error(f"GPU タスク失敗: {task.task_id}, エラー: {e}")
            
            with self.resource_lock:
                self.stats['failed_tasks'] += 1
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    def _cleanup_gpu_memory(self) -> None:
        """GPU メモリクリーンアップ"""
        try:
            if self.gpu_available:
                # PyTorchキャッシュをクリア
                torch.cuda.empty_cache()
                
                # ガベージコレクション
                gc.collect()
                
                with self.resource_lock:
                    self.stats['memory_cleanups'] += 1
                
                used_memory, usage_percent = self.get_memory_usage()
                self.logger.info(f"GPU メモリクリーンアップ実行: {used_memory}MB ({usage_percent:.1f}%)")
                
        except Exception as e:
            self.logger.error(f"GPU メモリクリーンアップエラー: {e}")
    
    def _monitor_gpu_usage(self) -> None:
        """GPU使用量監視スレッド"""
        while self.monitoring_active:
            try:
                used_memory, usage_percent = self.get_memory_usage()
                
                # 統計情報更新
                with self.resource_lock:
                    self.stats['peak_memory_usage'] = max(
                        self.stats['peak_memory_usage'], used_memory
                    )
                
                # 高使用率の場合は警告
                if usage_percent > 90:
                    self.logger.warning(f"GPU メモリ使用率が高い: {usage_percent:.1f}%")
                
                # 自動クリーンアップ
                if usage_percent > self.cleanup_threshold * 100:
                    self._cleanup_gpu_memory()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"GPU監視エラー: {e}")
                time.sleep(self.monitoring_interval)
    
    @contextmanager
    def managed_execution(self, task_id: str, priority: GPUPriority = GPUPriority.MEDIUM, 
                         estimated_memory: int = 100):
        """コンテキストマネージャーでリソース管理"""
        start_time = time.time()
        
        try:
            # メモリ使用量チェック
            if not self.is_memory_available(estimated_memory):
                self.logger.warning(f"メモリ不足: {task_id}")
                self._cleanup_gpu_memory()
                
                # 再チェック
                if not self.is_memory_available(estimated_memory):
                    raise RuntimeError(f"十分なGPUメモリがありません: {estimated_memory}MB必要")
            
            self.logger.debug(f"GPU リソース取得: {task_id}")
            yield
            
        except Exception as e:
            self.logger.error(f"GPU 実行エラー: {task_id}, {e}")
            raise
        
        finally:
            end_time = time.time()
            self.logger.debug(f"GPU リソース解放: {task_id} (実行時間: {end_time - start_time:.3f}秒)")
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        with self.resource_lock:
            used_memory, usage_percent = self.get_memory_usage()
            
            return {
                'gpu_available': self.gpu_available,
                'device_name': self.device_name,
                'total_memory_mb': self.total_memory,
                'used_memory_mb': used_memory,
                'usage_percent': usage_percent,
                'active_tasks': len(self.active_tasks),
                'queue_size': self.task_queue.qsize(),
                'stats': self.stats.copy()
            }
    
    def set_memory_threshold(self, threshold: float) -> None:
        """メモリ使用率の閾値を設定"""
        if 0.0 < threshold <= 1.0:
            self.memory_threshold = threshold
            self.logger.info(f"メモリ閾値を {threshold * 100:.1f}% に設定")
        else:
            self.logger.error("メモリ閾値は0.0-1.0の範囲で設定してください")
    
    def emergency_cleanup(self) -> None:
        """緊急時のメモリクリーンアップ"""
        self.logger.warning("緊急メモリクリーンアップ実行")
        
        # 全てのアクティブなタスクを一時停止
        with self.resource_lock:
            active_count = len(self.active_tasks)
        
        if active_count > 0:
            self.logger.warning(f"{active_count} 個のアクティブタスクがあります")
        
        # 強制的なメモリクリーンアップ
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()
        
        used_memory, usage_percent = self.get_memory_usage()
        self.logger.info(f"緊急クリーンアップ完了: {used_memory}MB ({usage_percent:.1f}%)")
    
    def shutdown(self) -> None:
        """リソース管理を終了"""
        try:
            self.logger.info("GPU リソース管理終了開始")
            
            # 監視スレッドを停止
            self.monitoring_active = False
            
            # 監視スレッドの終了を待機（タイムアウト付き）
            if self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=1.0)
                if self.monitoring_thread.is_alive():
                    self.logger.warning("監視スレッドの終了がタイムアウト")
            
            # エグゼキューターを終了（タイムアウト付き）
            try:
                self.gpu_executor.shutdown(wait=False)
                import time
                time.sleep(0.1)  # 短い待機
            except Exception as e:
                self.logger.warning(f"GPUエグゼキューター終了エラー: {e}")
            
            # 最終クリーンアップ
            try:
                self._cleanup_gpu_memory()
            except Exception as e:
                self.logger.warning(f"最終クリーンアップエラー: {e}")
            
            # 統計情報をログ出力
            try:
                stats = self.get_statistics()
                self.logger.info(f"GPU使用統計: {stats}")
            except Exception as e:
                self.logger.warning(f"統計情報取得エラー: {e}")
                
            self.logger.info("GPU リソース管理終了完了")
            
        except Exception as e:
            print(f"GPUリソースマネージャー終了エラー: {e}")

# グローバルインスタンス
_gpu_manager: Optional[GPUResourceManager] = None

def get_gpu_manager() -> GPUResourceManager:
    """GPU リソース管理のシングルトンインスタンスを取得"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUResourceManager()
    return _gpu_manager

def cleanup_gpu_manager() -> None:
    """GPU リソース管理をクリーンアップ"""
    global _gpu_manager
    if _gpu_manager is not None:
        _gpu_manager.shutdown()
        _gpu_manager = None