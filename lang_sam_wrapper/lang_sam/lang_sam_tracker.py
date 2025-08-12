"""統合Lang-SAM Trackerモジュール - 例外処理、トラッキング、フレームバッファリング機能を統合"""

import time
import cv2
import numpy as np
from collections import deque
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple, Any

from lang_sam.models.utils import DEVICE

# 前方宣言で循環インポートを回避
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lang_sam.models.coordinator import ModelCoordinator


# ========================================
# 例外クラス群 (exceptions.py統合)
# ========================================

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


# ========================================
# トラッキング機能群 (tracking.py統合)
# ========================================

class TrackingConfig:
    """トラッキング設定管理"""
    
    def __init__(self, bbox_margin: int = 5, bbox_min_size: int = 20, 
                 tracker_min_size: int = 10):
        self.bbox_margin = bbox_margin
        self.bbox_min_size = bbox_min_size
        self.tracker_min_size = tracker_min_size
    
    def update(self, **kwargs) -> None:
        """設定更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class CSRTTracker:
    """単一オブジェクト用CSRTトラッカー"""
    
    def __init__(self, tracker_id: str, bbox: Tuple[int, int, int, int], 
                 image: np.ndarray):
        self.tracker_id = tracker_id
        self.tracker = self._create_tracker()
        self.is_initialized = False
        
        if self.tracker and self._initialize(image, bbox):
            self.is_initialized = True
    
    def _create_tracker(self) -> Optional[cv2.Tracker]:
        """OpenCV CSRTトラッカー生成（バージョン対応）"""
        try:
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                return cv2.legacy.TrackerCSRT_create()
            else:
                return cv2.TrackerCSRT_create()
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "tracker creation", e)
    
    def _initialize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """トラッカー初期化"""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            tracker_bbox = (x1, y1, width, height)
            
            # BGR画像でトラッカー初期化
            bgr_image = self._ensure_bgr_format(image)
            return self.tracker.init(bgr_image, tracker_bbox)
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "initialization", e)
    
    def _ensure_bgr_format(self, image: np.ndarray) -> np.ndarray:
        """BGR形式確保"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    def update(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """追跡更新
        
        Returns:
            Optional[Tuple]: 更新されたbbox座標 (x1, y1, x2, y2) or None
        """
        if not self.is_initialized:
            return None
        
        try:
            bgr_image = self._ensure_bgr_format(image)
            success, bbox = self.tracker.update(bgr_image)
            
            if success and bbox is not None:
                x1, y1, w, h = [int(v) for v in bbox]
                return (x1, y1, x1 + w, y1 + h)
            return None
            
        except Exception as e:
            raise CSRTTrackingError(self.tracker_id, "update", e)


class TrackingManager:
    """複数オブジェクトCSRTトラッキング管理"""
    
    def __init__(self, tracking_targets: List[str], config: Optional[TrackingConfig] = None):
        self.tracking_targets = [target.lower() for target in tracking_targets]
        self.config = config or TrackingConfig()
        self.trackers: Dict[str, CSRTTracker] = {}
        self.tracked_boxes: Dict[str, List[float]] = {}
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        """追跡対象更新"""
        self.tracking_targets = [target.lower() for target in targets]
    
    def set_config(self, config: TrackingConfig) -> None:
        """設定更新"""
        self.config = config
    
    def initialize_trackers(self, boxes: np.ndarray, labels: List[str], image: np.ndarray) -> None:
        """複数トラッカー初期化"""
        try:
            # 既存トラッカー状態クリア
            self.clear_trackers()
            
            height, width = image.shape[:2]
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                # 追跡対象フィルタリング
                if not self._is_target_label(label):
                    continue
                
                # 座標変換・検証
                bbox = self._process_bbox(box, width, height)
                if bbox is None:
                    continue
                
                # トラッカー生成
                tracker_id = f"{label}_{i}"
                try:
                    tracker = CSRTTracker(tracker_id, bbox, image)
                    if tracker.is_initialized:
                        self.trackers[tracker_id] = tracker
                        self.tracked_boxes[tracker_id] = list(bbox)
                except CSRTTrackingError:
                    # 個別トラッカー初期化失敗は継続
                    continue
                    
        except Exception as e:
            raise CSRTTrackingError("batch", "initialization", e)
    
    def _is_target_label(self, label: str) -> bool:
        """追跡対象判定"""
        label_lower = label.lower()
        return any(target in label_lower for target in self.tracking_targets)
    
    def _process_bbox(self, box: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """BoundingBox座標処理・検証"""
        try:
            if len(box) < 4:
                return None
            
            # 正規化座標→ピクセル座標変換
            if np.max(box) <= 1.0:  # 正規化座標系
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
            else:  # ピクセル座標系
                x1, y1, x2, y2 = [int(v) for v in box]
            
            # 境界調整
            bbox = self._adjust_bbox_bounds(x1, y1, x2, y2, width, height)
            
            # サイズ検証
            if self._validate_bbox_size(bbox):
                return bbox
            return None
            
        except Exception:
            return None
    
    def _adjust_bbox_bounds(self, x1: int, y1: int, x2: int, y2: int, 
                           width: int, height: int) -> Tuple[int, int, int, int]:
        """BoundingBox境界調整"""
        margin = self.config.bbox_margin
        min_size = self.config.bbox_min_size
        
        x1 = max(margin, min(x1, width - margin - min_size))
        y1 = max(margin, min(y1, height - margin - min_size))
        x2 = max(x1 + min_size, min(x2, width - margin))
        y2 = max(y1 + min_size, min(y2, height - margin))
        
        return (x1, y1, x2, y2)
    
    def _validate_bbox_size(self, bbox: Tuple[int, int, int, int]) -> bool:
        """BoundingBoxサイズ検証"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width >= self.config.bbox_min_size and height >= self.config.bbox_min_size
    
    def update_all_trackers(self, image: np.ndarray) -> Dict[str, List[float]]:
        """全トラッカー更新"""
        if not self.trackers:
            return {}
        
        height, width = image.shape[:2]
        failed_trackers = []
        
        for tracker_id, tracker in list(self.trackers.items()):
            try:
                updated_bbox = tracker.update(image)
                
                if updated_bbox is not None:
                    # クリッピング処理
                    clipped_bbox = self._clip_bbox_to_image(updated_bbox, width, height)
                    
                    if self._validate_clipped_bbox(clipped_bbox):
                        self.tracked_boxes[tracker_id] = list(clipped_bbox)
                    else:
                        failed_trackers.append(tracker_id)
                else:
                    failed_trackers.append(tracker_id)
                    
            except CSRTTrackingError:
                failed_trackers.append(tracker_id)
        
        # 失敗トラッカー削除
        for tracker_id in failed_trackers:
            self._remove_tracker(tracker_id)
        
        return self.tracked_boxes.copy()
    
    def _clip_bbox_to_image(self, bbox: Tuple[int, int, int, int], 
                           width: int, height: int) -> Tuple[int, int, int, int]:
        """画像境界内BoundingBoxクリッピング"""
        x1, y1, x2, y2 = bbox
        x1_clipped = max(0, min(x1, width - 1))
        y1_clipped = max(0, min(y1, height - 1))
        x2_clipped = max(x1_clipped + 1, min(x2, width))
        y2_clipped = max(y1_clipped + 1, min(y2, height))
        return (x1_clipped, y1_clipped, x2_clipped, y2_clipped)
    
    def _validate_clipped_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """クリップ後BoundingBox検証"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width > self.config.tracker_min_size and height > self.config.tracker_min_size
    
    def _remove_tracker(self, tracker_id: str) -> None:
        """トラッカー削除"""
        self.trackers.pop(tracker_id, None)
        self.tracked_boxes.pop(tracker_id, None)
    
    def clear_trackers(self) -> None:
        """全トラッカー状態クリア"""
        self.trackers.clear()
        self.tracked_boxes.clear()
    
    def get_tracking_result(self) -> Dict[str, Any]:
        """追跡結果取得"""
        if not self.tracked_boxes:
            return {
                "boxes": np.array([]),
                "labels": [],
                "scores": np.array([])
            }
        
        boxes = list(self.tracked_boxes.values())
        labels = list(self.tracked_boxes.keys())
        scores = np.ones(len(boxes))
        
        return {
            "boxes": np.array(boxes),
            "labels": labels,
            "scores": scores
        }
    
    def has_active_trackers(self) -> bool:
        """アクティブトラッカー存在確認"""
        return len(self.trackers) > 0


# ========================================
# フレームバッファリング機能群 (frame_buffer.py統合)
# ========================================

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


# ========================================
# メインLangSAMTrackerクラス
# ========================================

class LangSAMTracker:
    """Language Segment-Anything + CSRT統合トラッカー    
    責任:
    - 高レベルなAPI提供
    - モデルコーディネーターへの委譲
    - 設定管理
    
    複雑な実装詳細は各専門クラスに委譲:
    - ModelCoordinator: モデル統合・パイプライン実行
    - TrackingManager: CSRT追跡管理
    - SafeImageProcessor: 画像処理・検証
    """
    
    def __init__(self, sam_type: str = "sam2.1_hiera_small", 
                 ckpt_path: Optional[str] = None, device=DEVICE):
        """LangSAMTracker初期化"""
        self.sam_type = sam_type
        
        # 循環インポートを回避するためにランタイムインポート
        from lang_sam.models.coordinator import ModelCoordinator
        
        # モデルコーディネーター初期化（主要な複雑ロジックを委譲）
        self.coordinator = ModelCoordinator(sam_type, ckpt_path, device)
        
        # デフォルト追跡設定
        self._default_tracking_targets = ["white line", "red pylon", "human", "car"]
        self._setup_default_tracking()
    
    def _setup_default_tracking(self) -> None:
        """デフォルトトラッキング設定"""
        default_config = {
            'bbox_margin': 5,
            'bbox_min_size': 20,
            'tracker_min_size': 10
        }
        self.coordinator.setup_tracking(self._default_tracking_targets, default_config)
    
    def predict_with_tracking(
        self,
        images_pil: List[Image.Image],
        texts_prompt: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        update_trackers: bool = True,
        run_sam: bool = True
    ) -> List[Dict]:
        """統合推論パイプライン（GroundingDINO → CSRT → SAM2）
        
        複雑な実装詳細はModelCoordinatorに委譲
        """
        try:
            return self.coordinator.predict_full_pipeline(
                images_pil, texts_prompt, box_threshold, text_threshold,
                update_trackers, run_sam
            )
        except LangSAMError:
            # カスタム例外は再発生
            raise
        except Exception as e:
            # 一般例外をラップ
            raise ErrorHandler.handle_model_error("full_pipeline", e)
    
    def update_trackers_only(self, image_np: np.ndarray) -> Dict:
        """CSRT追跡のみ実行（高速版）
        
        トラッキング詳細はTrackingManagerに委譲
        """
        try:
            return self.coordinator.update_tracking_only(image_np)
        except LangSAMError:
            raise
        except Exception as e:
            raise ErrorHandler.handle_tracking_error("batch", "update_only", e)
    
    def update_trackers_with_sam(self, image_np: np.ndarray) -> Dict:
        """CSRT追跡 + SAM2セグメンテーション統合実行
        
        複雑な統合ロジックはModelCoordinatorに委譲
        """
        try:
            return self.coordinator.update_tracking_with_sam(image_np)
        except LangSAMError:
            raise
        except Exception as e:
            raise ErrorHandler.handle_model_error("tracking_with_sam", e)
    
    def set_tracking_targets(self, targets: List[str]) -> None:
        """追跡対象ラベル設定
        
        設定管理はTrackingManagerに委譲
        """
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_manager.set_tracking_targets(targets)
    
    def set_tracking_config(self, config: Dict[str, int]) -> None:
        """トラッキング設定更新
        
        設定管理はTrackingConfigに委譲
        """
        if self.coordinator.tracking_manager:
            self.coordinator.tracking_config.update(**config)
    
    def clear_trackers(self) -> None:
        """全トラッカー状態クリア
        
        状態管理はModelCoordinatorに委譲
        """
        self.coordinator.clear_tracking_state()
    
    # 互換性維持のためのプロパティ
    @property
    def has_active_tracking(self) -> bool:
        """アクティブ追跡状態確認"""
        return self.coordinator.has_active_tracking()
    
    @property 
    def tracking_targets(self) -> List[str]:
        """現在の追跡対象取得"""
        if self.coordinator.tracking_manager:
            return self.coordinator.tracking_manager.tracking_targets
        return self._default_tracking_targets
    
    @property
    def config(self) -> Dict[str, int]:
        """現在のトラッキング設定取得"""
        config_obj = self.coordinator.tracking_config
        return {
            'bbox_margin': config_obj.bbox_margin,
            'bbox_min_size': config_obj.bbox_min_size,
            'tracker_min_size': config_obj.tracker_min_size
        }