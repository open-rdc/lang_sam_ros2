"""フレームバッファリング - GroundingDINO遅延補正システム"""

import time
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from lang_sam.exceptions import LangSAMError


class FrameBuffer:
    """GroundingDINO処理中のフレーム蓄積と高速キャッチアップ"""
    
    def __init__(self, max_buffer_seconds: float = 5.0):
        """
        Args:
            max_buffer_seconds: 最大バッファ時間（秒）
        """
        self.max_buffer_seconds = max_buffer_seconds
        self.max_frames = int(max_buffer_seconds * 30)  # 30fps想定
        
        # バッファ状態管理
        self.is_buffering = False
        self.gdino_start_time = 0.0
        
        # フレームバッファ（時系列順）
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
        # 統計情報
        self.total_buffered_frames = 0
        self.last_catchup_duration = 0.0
    
    def start_gdino_processing(self) -> None:
        """GroundingDINO処理開始 - バッファリング開始"""
        self.is_buffering = True
        self.gdino_start_time = time.time()
        self.frames.clear()
        self.timestamps.clear()
        self.total_buffered_frames = 0
    
    def add_frame(self, frame: np.ndarray) -> None:
        """フレーム追加（バッファリング中のみ）"""
        if not self.is_buffering:
            return
        
        current_time = time.time()
        
        # フレームをコピーして蓄積（参照渡しを避ける）
        self.frames.append(frame.copy())
        self.timestamps.append(current_time)
        self.total_buffered_frames += 1
    
    def finish_gdino_and_catchup(self, detection_result: Dict[str, Any], 
                                current_tracker_manager) -> Dict[str, Any]:
        """GroundingDINO完了 - 高速キャッチアップ実行
        
        Args:
            detection_result: GroundingDINOの検出結果
            current_tracker_manager: トラッキングマネージャー
            
        Returns:
            最新フレームでの追跡結果
        """
        if not self.is_buffering:
            return detection_result
        
        self.is_buffering = False
        catchup_start = time.time()
        
        try:
            # バッファされたフレームがない場合は通常処理
            if len(self.frames) == 0:
                return detection_result
            
            # 検出結果でトラッカー初期化（最初のバッファフレーム）
            first_frame = self.frames[0]
            boxes = detection_result.get("boxes", [])
            labels = detection_result.get("labels", [])
            
            if len(boxes) == 0:
                return detection_result
            
            # トラッカー初期化
            current_tracker_manager.initialize_trackers(boxes, labels, first_frame)
            
            # 蓄積フレームで高速追跡実行
            final_result = self._fast_forward_tracking(current_tracker_manager)
            
            # 統計更新
            self.last_catchup_duration = time.time() - catchup_start
            
            return final_result
            
        except Exception as e:
            # エラー時は元の結果を返却
            return detection_result
    
    def _fast_forward_tracking(self, tracker_manager) -> Dict[str, Any]:
        """蓄積フレームでの高速追跡実行"""
        if len(self.frames) <= 1:
            return tracker_manager.get_tracking_result()
        
        # 2フレーム目以降で高速追跡
        for i in range(1, len(self.frames)):
            frame = self.frames[i]
            tracker_manager.update_all_trackers(frame)
        
        # 最終追跡結果返却
        return tracker_manager.get_tracking_result()
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """バッファ統計情報取得"""
        return {
            "is_buffering": self.is_buffering,
            "buffered_frames": len(self.frames),
            "total_buffered": self.total_buffered_frames,
            "last_catchup_duration": self.last_catchup_duration,
            "buffer_timespan": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) >= 2 else 0.0
        }
    
    def get_frame_at_time_offset(self, seconds_ago: float) -> Optional[Tuple[np.ndarray, float]]:
        """指定時間前のフレーム取得（時間さかのぼり）
        
        Args:
            seconds_ago: 何秒前のフレームを取得するか
            
        Returns:
            Optional[Tuple]: (frame, timestamp) or None if not found
        """
        if len(self.timestamps) == 0:
            return None
        
        current_time = time.time()
        target_time = current_time - seconds_ago
        
        # 最も近いタイムスタンプのフレーム検索
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
        
        # 時間範囲内のフレーム収集
        sequence = []
        for i, timestamp in enumerate(self.timestamps):
            if start_time <= timestamp <= end_time:
                sequence.append((self.frames[i].copy(), timestamp))
        
        # 時系列ソート
        sequence.sort(key=lambda x: x[1])
        return sequence
    
    def clear_buffer(self) -> None:
        """バッファクリア"""
        self.is_buffering = False
        self.frames.clear()
        self.timestamps.clear()
        self.total_buffered_frames = 0


class RealtimeFrameManager:
    """リアルタイムフレーム管理とバッファリング統合"""
    
    def __init__(self, buffer_duration: float = 5.0):
        self.frame_buffer = FrameBuffer(buffer_duration)
        self.latest_frame = None
        self.frame_count = 0
    
    def process_incoming_frame(self, frame: np.ndarray) -> None:
        """受信フレーム処理"""
        self.latest_frame = frame
        self.frame_count += 1
        
        # バッファリング中の場合は蓄積
        self.frame_buffer.add_frame(frame)
    
    def start_gdino_processing(self) -> None:
        """GroundingDINO処理開始通知"""
        self.frame_buffer.start_gdino_processing()
    
    def complete_gdino_processing(self, detection_result: Dict[str, Any], 
                                 tracker_manager) -> Dict[str, Any]:
        """GroundingDINO処理完了とキャッチアップ"""
        return self.frame_buffer.finish_gdino_and_catchup(detection_result, tracker_manager)
    
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
    
    def get_past_frame(self, seconds_ago: float) -> Optional[Tuple[np.ndarray, float]]:
        """過去フレーム取得（時間さかのぼり）"""
        return self.frame_buffer.get_frame_at_time_offset(seconds_ago)
    
    def get_frame_history(self, duration_seconds: float) -> List[Tuple[np.ndarray, float]]:
        """指定期間のフレーム履歴取得"""
        return self.frame_buffer.get_frame_sequence(duration_seconds, 0.0)