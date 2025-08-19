"""CSRTトラッキング用フレームバッファー実装

フレーム履歴の管理と時間遡行復旧機能を提供します。
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from .exceptions import TrackingError, ErrorHandler


@dataclass
class FrameData:
    """フレームデータ構造"""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    metadata: Optional[Dict] = None


class CSRTFrameBuffer:
    """CSRTトラッキング用フレームバッファ
    
    指定された時間だけフレーム履歴を保持し、
    時間遡行による復旧機能を提供します。
    """
    
    def __init__(self, buffer_duration: float = 5.0):
        """
        Args:
            buffer_duration: バッファ保持時間（秒）
        """
        self.buffer_duration = buffer_duration
        self.frames: List[FrameData] = []
        self.current_frame_id = 0
        
    def add_frame(self, frame: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """フレームをバッファに追加
        
        Args:
            frame: 入力フレーム
            metadata: フレームメタデータ
            
        Returns:
            追加されたフレームのID
        """
        current_time = time.time()
        frame_data = FrameData(
            frame=frame.copy(),
            timestamp=current_time,
            frame_id=self.current_frame_id,
            metadata=metadata
        )
        
        self.frames.append(frame_data)
        self.current_frame_id += 1
        
        # 古いフレームを削除
        self._cleanup_old_frames(current_time)
        
        return frame_data.frame_id
        
    def get_frame_by_id(self, frame_id: int) -> Optional[FrameData]:
        """フレームIDでフレームを取得"""
        for frame_data in self.frames:
            if frame_data.frame_id == frame_id:
                return frame_data
        return None
        
    def get_frames_in_time_range(self, start_time: float, end_time: float) -> List[FrameData]:
        """指定時間範囲のフレームを取得"""
        return [
            frame_data for frame_data in self.frames
            if start_time <= frame_data.timestamp <= end_time
        ]
        
    def get_recent_frames(self, count: int) -> List[FrameData]:
        """最新のN個のフレームを取得"""
        return self.frames[-count:] if count <= len(self.frames) else self.frames[:]
        
    def get_frame_time_ago(self, seconds_ago: float) -> Optional[FrameData]:
        """指定秒前のフレームを取得"""
        if not self.frames:
            return None
            
        current_time = time.time()
        target_time = current_time - seconds_ago
        
        # 最も近い時刻のフレームを見つける
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data in self.frames:
            time_diff = abs(frame_data.timestamp - target_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_frame = frame_data
                
        return closest_frame
        
    def _cleanup_old_frames(self, current_time: float):
        """古いフレームを削除"""
        cutoff_time = current_time - self.buffer_duration
        self.frames = [
            frame_data for frame_data in self.frames
            if frame_data.timestamp >= cutoff_time
        ]
        
    def get_buffer_info(self) -> Dict:
        """バッファ情報を取得"""
        return {
            'frame_count': len(self.frames),
            'buffer_duration': self.buffer_duration,
            'oldest_timestamp': self.frames[0].timestamp if self.frames else None,
            'newest_timestamp': self.frames[-1].timestamp if self.frames else None,
            'current_frame_id': self.current_frame_id
        }


class CSRTFrameManager:
    """CSRT復旧機能付きフレーム管理
    
    フレームバッファと時間遡行復旧を統合管理します。
    """
    
    def __init__(self, 
                 buffer_duration: float = 5.0,
                 time_travel_seconds: float = 1.0,
                 fast_forward_frames: int = 10):
        """
        Args:
            buffer_duration: フレームバッファ保持時間
            time_travel_seconds: 時間遡行可能秒数
            fast_forward_frames: 早送り処理フレーム数
        """
        self.frame_buffer = CSRTFrameBuffer(buffer_duration)
        self.time_travel_seconds = time_travel_seconds
        self.fast_forward_frames = fast_forward_frames
        self.error_handler = ErrorHandler()
        
        # 復旧状態
        self.recovery_mode = False
        self.recovery_start_time = None
        
    def add_frame(self, frame: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """フレームを追加"""
        return self.frame_buffer.add_frame(frame, metadata)
        
    def get_recovery_frame(self) -> Optional[FrameData]:
        """復旧用フレームを取得（時間遡行）"""
        try:
            recovery_frame = self.frame_buffer.get_frame_time_ago(self.time_travel_seconds)
            if recovery_frame is None:
                self.error_handler.handle_error(
                    TrackingError("復旧用フレームが見つかりません"),
                    context="frame_recovery"
                )
                return None
                
            return recovery_frame
            
        except Exception as e:
            self.error_handler.handle_error(
                TrackingError(f"復旧フレーム取得エラー: {str(e)}"),
                context="frame_recovery"
            )
            return None
            
    def start_recovery_mode(self):
        """復旧モード開始"""
        self.recovery_mode = True
        self.recovery_start_time = time.time()
        
    def stop_recovery_mode(self):
        """復旧モード終了"""
        self.recovery_mode = False
        self.recovery_start_time = None
        
    def get_fast_forward_frames(self) -> List[FrameData]:
        """早送り用フレームリストを取得"""
        return self.frame_buffer.get_recent_frames(self.fast_forward_frames)
        
    def is_in_recovery_mode(self) -> bool:
        """復旧モード中かどうか"""
        return self.recovery_mode
        
    def get_recovery_duration(self) -> Optional[float]:
        """復旧モード継続時間"""
        if not self.recovery_mode or self.recovery_start_time is None:
            return None
        return time.time() - self.recovery_start_time
        
    def get_manager_status(self) -> Dict:
        """マネージャー状態を取得"""
        buffer_info = self.frame_buffer.get_buffer_info()
        return {
            **buffer_info,
            'recovery_mode': self.recovery_mode,
            'recovery_duration': self.get_recovery_duration(),
            'time_travel_seconds': self.time_travel_seconds,
            'fast_forward_frames': self.fast_forward_frames
        }