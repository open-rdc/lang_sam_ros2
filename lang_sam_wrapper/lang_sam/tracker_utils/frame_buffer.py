"""CSRTフレームバッファリング機能"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


class CSRTFrameBuffer:
    """CSRT専用フレームバッファ - 画像保存、時間さかのぼり、早送り機能"""
    
    def __init__(self, max_buffer_seconds: float = 5.0):
        """
        Args:
            max_buffer_seconds: 最大バッファ時間（秒）
        """
        self.max_buffer_seconds = max_buffer_seconds
        self.max_frames = int(max_buffer_seconds * 30)  # 30fps想定
        
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
        self.total_buffered_frames = 0
        self.last_catchup_duration = 0.0
    
    def add_frame(self, frame: np.ndarray) -> None:
        """フレーム追加（常時蓄積）"""
        current_time = time.time()
        
        self.frames.append(frame.copy())
        self.timestamps.append(current_time)
        self.total_buffered_frames += 1
    
    def fast_forward_tracking(self, tracker_manager, start_frame_idx: int = 0) -> Dict[str, Any]:
        """蓄積フレームでの高速追跡実行（早送り機能）"""
        if len(self.frames) <= start_frame_idx + 1:
            return tracker_manager.get_tracking_result()
        
        for i in range(start_frame_idx + 1, len(self.frames)):
            frame = self.frames[i]
            tracker_manager.update_all_trackers(frame)
        
        return tracker_manager.get_tracking_result()
    
    def get_frame_at_time_offset(self, seconds_ago: float) -> Optional[Tuple[np.ndarray, float]]:
        """指定時間前のフレーム取得（時間さかのぼり機能）
        
        Args:
            seconds_ago: 何秒前のフレームを取得するか
            
        Returns:
            Optional[Tuple]: (frame, timestamp) or None if not found
        """
        if len(self.timestamps) == 0:
            return None
        
        current_time = time.time()
        target_time = current_time - seconds_ago
        
        best_idx = -1
        min_diff = float('inf')
        
        for i, timestamp in enumerate(self.timestamps):
            time_diff = abs(timestamp - target_time)
            if time_diff < min_diff:
                min_diff = time_diff
                best_idx = i
        
        if best_idx >= 0:
            return self.frames[best_idx].copy(), self.timestamps[best_idx]
        return None
    
    def get_frame_sequence(self, start_seconds_ago: float, 
                          end_seconds_ago: float = 0.0) -> List[Tuple[np.ndarray, float]]:
        """指定期間のフレーム系列取得（時間範囲さかのぼり）
        
        Args:
            start_seconds_ago: 開始時間（秒前）
            end_seconds_ago: 終了時間（秒前、デフォルト0=現在）
            
        Returns:
            List[Tuple]: [(frame, timestamp), ...] 時系列順
        """
        if len(self.timestamps) == 0:
            return []
        
        current_time = time.time()
        start_time = current_time - start_seconds_ago
        end_time = current_time - end_seconds_ago
        
        sequence = []
        for i, timestamp in enumerate(self.timestamps):
            if start_time <= timestamp <= end_time:
                sequence.append((self.frames[i].copy(), timestamp))
        
        sequence.sort(key=lambda x: x[1])
        return sequence
    
    def get_latest_frames(self, count: int = 5) -> List[Tuple[np.ndarray, float]]:
        """最新N個のフレーム取得"""
        if len(self.frames) == 0:
            return []
        
        start_idx = max(0, len(self.frames) - count)
        result = []
        for i in range(start_idx, len(self.frames)):
            result.append((self.frames[i].copy(), self.timestamps[i]))
        
        return result
    
    def clear_buffer(self) -> None:
        """バッファクリア"""
        self.frames.clear()
        self.timestamps.clear()
        self.total_buffered_frames = 0
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """バッファ統計情報取得"""
        return {
            "buffered_frames": len(self.frames),
            "total_buffered": self.total_buffered_frames,
            "last_catchup_duration": self.last_catchup_duration,
            "buffer_timespan": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) >= 2 else 0.0
        }


class CSRTFrameManager:
    """CSRT専用フレーム管理 - 画像保存、時間操作、キャッチアップ機能統合"""
    
    def __init__(self, buffer_duration: float = 5.0):
        self.frame_buffer = CSRTFrameBuffer(buffer_duration)
        self.latest_frame = None
        self.frame_count = 0
    
    def process_incoming_frame(self, frame: np.ndarray) -> None:
        """受信フレーム処理（常時蓄積）"""
        self.latest_frame = frame
        self.frame_count += 1
        
        self.frame_buffer.add_frame(frame)
    
    def fast_forward_csrt_tracking(self, tracker_manager, frames_to_skip: int = 0) -> Dict[str, Any]:
        """CSRT高速キャッチアップ（早送り機能）"""
        return self.frame_buffer.fast_forward_tracking(tracker_manager, frames_to_skip)
    
    def get_past_frame(self, seconds_ago: float) -> Optional[Tuple[np.ndarray, float]]:
        """過去フレーム取得（時間さかのぼり）"""
        return self.frame_buffer.get_frame_at_time_offset(seconds_ago)
    
    def get_frame_history(self, duration_seconds: float) -> List[Tuple[np.ndarray, float]]:
        """指定期間のフレーム履歴取得"""
        return self.frame_buffer.get_frame_sequence(duration_seconds, 0.0)
    
    def get_recent_frames(self, count: int = 5) -> List[Tuple[np.ndarray, float]]:
        """最新N個のフレーム取得"""
        return self.frame_buffer.get_latest_frames(count)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """最新フレーム取得"""
        return self.latest_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        buffer_stats = self.frame_buffer.get_buffer_stats()
        return {
            **buffer_stats,
            "total_frames_processed": self.frame_count,
            "latest_frame_available": self.latest_frame is not None
        }
    
    def clear_history(self) -> None:
        """履歴クリア"""
        self.frame_buffer.clear_buffer()