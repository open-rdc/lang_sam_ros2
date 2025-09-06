#!/usr/bin/env python3
"""
LangSAM Tracker - OpticalFlow版
GroundingDINO + SAM2 + OpticalFlow特徴点追跡統合システム
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple, Union
import time

# LangSAMコンポーネントインポート
from .models import GroundingDINO, SAM2


class OpticalFlowTracker:
    """特徴点ベースOptical Flowトラッカー"""
    
    def __init__(self, 
                 max_corners: int = 100,
                 quality_level: float = 0.01,
                 min_distance: int = 10,
                 block_size: int = 7,
                 win_size: Tuple[int, int] = (15, 15),
                 max_level: int = 2,
                 max_disappeared: int = 30,
                 min_tracked_points: int = 5):
        """
        初期化
        Args:
            max_corners: 抽出する最大特徴点数
            quality_level: 特徴点の品質レベル閾値
            min_distance: 特徴点間の最小距離
            block_size: 特徴点検出のブロックサイズ
            win_size: Optical Flowの探索ウィンドウサイズ
            max_level: ピラミッドレベル数
            max_disappeared: トラック削除までの最大未検出フレーム数
            min_tracked_points: トラック維持に必要な最小特徴点数
        """
        # 特徴点抽出パラメータ（goodFeaturesToTrack用）
        # 目的: 安定した特徴点を抽出する目的でパラメータを設定
        self.feature_params = {
            'maxCorners': max_corners,
            'qualityLevel': quality_level,
            'minDistance': min_distance,
            'blockSize': block_size
        }
        
        # Lucas-Kanade Optical Flowパラメータ
        # 目的: 高精度な特徴点追跡を実現する目的でパラメータを設定
        self.lk_params = {
            'winSize': win_size,
            'maxLevel': max_level,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # トラッカー管理パラメータ
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # {track_id: {"points": np.array, "bbox": [x1,y1,x2,y2], "label": str, "disappeared": int}}
        self.max_disappeared = max_disappeared
        self.min_tracked_points = min_tracked_points
        
        # 前フレーム保存用（グレースケール）
        self.prev_gray: Optional[np.ndarray] = None
        
        print(f"[OpticalFlowTracker] 初期化完了: max_corners={max_corners}, min_tracked_points={min_tracked_points}")
    
    def update(self, 
              frame: np.ndarray,
              detections: Union[List, np.ndarray], 
              labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Optical Flowトラッカー更新
        Args:
            frame: 現在のフレーム画像（BGR）
            detections: 検出結果 [[x1, y1, x2, y2, score], ...] または空リスト
            labels: 検出ラベルリスト
        Returns:
            tracks: トラック結果 [[x1, y1, x2, y2, track_id], ...]
            track_labels: {track_id: label}のマッピング
        """
        # グレースケール変換
        # 目的: Optical Flow計算のためグレースケール化する目的で使用
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 初回フレームまたは前フレームがない場合
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            # 検出結果から初期トラックを作成
            if len(detections) > 0:
                self._initialize_tracks_from_detections(gray, detections, labels)
            return self._get_current_tracks()
        
        # 既存トラックの特徴点を追跡
        # 目的: calcOpticalFlowPyrLKで特徴点の動きを追跡する目的で使用
        self._track_existing_objects(gray)
        
        # 新しい検出結果の処理
        if len(detections) > 0:
            self._process_new_detections(gray, detections, labels)
        
        # disappeared countが閾値を超えたトラックを削除
        self._cleanup_tracks()
        
        # 前フレームを更新
        self.prev_gray = gray.copy()
        
        return self._get_current_tracks()
    
    def _initialize_tracks_from_detections(self, 
                                          gray: np.ndarray, 
                                          detections: Union[List, np.ndarray],
                                          labels: Optional[List[str]]):
        """検出結果から初期トラックを作成"""
        # 目的: 最初のフレームで検出されたオブジェクトの特徴点を抽出する目的で使用
        if isinstance(detections, (list, np.ndarray)):
            dets = np.asarray(detections, dtype=np.float32)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
        else:
            return
        
        for i, det in enumerate(dets):
            bbox = det[:4].astype(int)
            x1, y1, x2, y2 = bbox
            
            # BBOX内の領域を切り出し
            # 目的: BBOX内のみから特徴点を抽出する目的で使用
            roi = gray[y1:y2, x1:x2]
            
            # 特徴点を検出
            # 目的: goodFeaturesToTrackでコーナー特徴点を検出する目的で使用
            corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)
            
            if corners is not None and len(corners) >= self.min_tracked_points:
                # ROI座標を画像座標に変換
                corners[:, 0, 0] += x1
                corners[:, 0, 1] += y1
                
                # 新しいトラックを登録
                track_id = self.next_id
                self.tracks[track_id] = {
                    "points": corners,
                    "bbox": bbox.tolist(),
                    "label": labels[i] if labels and i < len(labels) else "unknown",
                    "disappeared": 0
                }
                self.next_id += 1
                print(f"[OpticalFlowTracker] 新規トラック{track_id}を登録: {len(corners)}個の特徴点")
    
    def _track_existing_objects(self, gray: np.ndarray):
        """既存トラックの特徴点を追跡"""
        # 目的: calcOpticalFlowPyrLKで前フレームから現フレームへの特徴点移動を計算する目的で使用
        
        for track_id, track in list(self.tracks.items()):
            old_points = track["points"]
            
            if old_points is None or len(old_points) == 0:
                track["disappeared"] += 1
                continue
            
            # Optical Flowで特徴点を追跡
            # 目的: Lucas-Kanade法で特徴点の新しい位置を計算する目的で使用
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, old_points, None, **self.lk_params
            )
            
            # 追跡成功した特徴点のみを保持
            if status is not None:
                good_new = new_points[status == 1]
                good_old = old_points[status == 1]
                
                if len(good_new) >= self.min_tracked_points:
                    # 特徴点群の移動ベクトルを計算
                    # 目的: 全体の動きを推定してBBOXを更新する目的で使用
                    movement = np.median(good_new - good_old, axis=0)
                    
                    # BBOXを更新
                    old_bbox = track["bbox"]
                    new_bbox = [
                        old_bbox[0] + movement[0],
                        old_bbox[1] + movement[1],
                        old_bbox[2] + movement[0],
                        old_bbox[3] + movement[1]
                    ]
                    
                    # 画像境界内にクリップ
                    h, w = gray.shape
                    new_bbox[0] = max(0, min(new_bbox[0], w-1))
                    new_bbox[1] = max(0, min(new_bbox[1], h-1))
                    new_bbox[2] = max(new_bbox[0]+1, min(new_bbox[2], w))
                    new_bbox[3] = max(new_bbox[1]+1, min(new_bbox[3], h))
                    
                    # トラック情報を更新
                    track["points"] = good_new.reshape(-1, 1, 2)
                    track["bbox"] = new_bbox
                    track["disappeared"] = 0
                    
                    print(f"[OpticalFlowTracker] トラック{track_id}更新: {len(good_new)}個の特徴点を追跡")
                else:
                    # 追跡できた特徴点が少なすぎる
                    track["disappeared"] += 1
                    track["points"] = None
                    print(f"[OpticalFlowTracker] トラック{track_id}: 特徴点不足 ({len(good_new)}個)")
            else:
                track["disappeared"] += 1
                track["points"] = None
    
    def _process_new_detections(self, 
                               gray: np.ndarray,
                               detections: Union[List, np.ndarray],
                               labels: Optional[List[str]]):
        """新しい検出結果を処理し、既存トラックと照合または新規登録"""
        # 目的: 新規検出を既存トラックと照合し、マッチしない場合は新規登録する目的で使用
        
        if isinstance(detections, (list, np.ndarray)):
            dets = np.asarray(detections, dtype=np.float32)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
        else:
            return
        
        # 各検出に対して処理
        for i, det in enumerate(dets):
            bbox = det[:4].astype(int)
            x1, y1, x2, y2 = bbox
            
            # 既存トラックとのIoUを計算して照合
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track["bbox"] is not None:
                    iou = self._calculate_iou(bbox, track["bbox"])
                    if iou > best_iou and iou > 0.3:  # IoU閾値
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # 既存トラックを更新（特徴点を再抽出）
                # 目的: 検出でトラックが確認されたので特徴点を再初期化する目的で使用
                roi = gray[y1:y2, x1:x2]
                corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)
                
                if corners is not None and len(corners) >= self.min_tracked_points:
                    corners[:, 0, 0] += x1
                    corners[:, 0, 1] += y1
                    self.tracks[best_track_id]["points"] = corners
                    self.tracks[best_track_id]["bbox"] = bbox.tolist()
                    self.tracks[best_track_id]["disappeared"] = 0
                    print(f"[OpticalFlowTracker] トラック{best_track_id}を再初期化: {len(corners)}個の特徴点")
            else:
                # 新規トラックとして登録
                roi = gray[y1:y2, x1:x2]
                corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)
                
                if corners is not None and len(corners) >= self.min_tracked_points:
                    corners[:, 0, 0] += x1
                    corners[:, 0, 1] += y1
                    
                    track_id = self.next_id
                    self.tracks[track_id] = {
                        "points": corners,
                        "bbox": bbox.tolist(),
                        "label": labels[i] if labels and i < len(labels) else "unknown",
                        "disappeared": 0
                    }
                    self.next_id += 1
                    print(f"[OpticalFlowTracker] 新規トラック{track_id}を登録: {len(corners)}個の特徴点")
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """IoU（Intersection over Union）を計算"""
        # 目的: 2つのBBOXの重なり度合いを計算する目的で使用
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 交差領域の計算
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _cleanup_tracks(self):
        """disappeared countが閾値を超えたトラックを削除"""
        # 目的: 長時間検出されないトラックを削除する目的で使用
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["disappeared"] > self.max_disappeared:
                del self.tracks[track_id]
                print(f"[OpticalFlowTracker] トラック{track_id}を削除")
    
    def _get_current_tracks(self) -> Tuple[np.ndarray, Dict[str, str]]:
        """現在のトラック情報を取得"""
        # 目的: 有効なトラック情報を整形して返す目的で使用
        if len(self.tracks) == 0:
            return np.empty((0, 5), dtype=np.float32), {}
        
        tracks_list = []
        track_labels = {}
        
        for track_id, track in self.tracks.items():
            if track["bbox"] is not None and track["disappeared"] == 0:
                bbox = track["bbox"]
                tracks_list.append([bbox[0], bbox[1], bbox[2], bbox[3], float(track_id)])
                track_labels[track_id] = track["label"]
        
        if len(tracks_list) == 0:
            return np.empty((0, 5), dtype=np.float32), {}
        
        return np.array(tracks_list, dtype=np.float32), track_labels
    
    def reset(self) -> None:
        """トラッカーリセット"""
        # 目的: 新規トラッキングセッション開始時に状態をクリアする目的で使用
        self.tracks.clear()
        self.next_id = 1
        self.prev_gray = None
        print("[OpticalFlowTracker] リセット完了")


class LangSamTracker:
    """LangSAM統合トラッカー（OpticalFlow Tracker版）"""
    def __init__(self,
                 sam_model: str = "sam2.1_hiera_tiny",
                 gdino_model: str = "gdino_tiny",
                 device: str = "cuda",
                 # OpticalFlowTrackerパラメータ
                 optical_flow_max_corners: int = 100,
                 optical_flow_quality_level: float = 0.01,
                 optical_flow_min_distance: int = 10,
                 optical_flow_block_size: int = 7,
                 optical_flow_win_size: Tuple[int, int] = (15, 15),
                 optical_flow_max_level: int = 2,
                 optical_flow_max_disappeared: int = 30,
                 optical_flow_min_tracked_points: int = 5):
        """初期化（OpticalFlow Tracker版）"""
        # GroundingDINO初期化
        # 目的: ゼロショット物体検出を実現する目的で使用
        self.gdino = GroundingDINO()
        self.gdino.build_model(device=device)
        
        # SAM2初期化
        # 目的: 高精度なセグメンテーションを実現する目的で使用
        self.sam = SAM2()
        self.sam.build_model(sam_model, device=device)
        
        # OpticalFlow Tracker初期化（特徴点追跡）
        # 目的: goodFeaturesToTrack + calcOpticalFlowPyrLKによる堅牢なトラッキングを実現する目的で使用
        self.optical_tracker = OpticalFlowTracker(
            max_corners=optical_flow_max_corners,
            quality_level=optical_flow_quality_level,
            min_distance=optical_flow_min_distance,
            block_size=optical_flow_block_size,
            win_size=optical_flow_win_size,
            max_level=optical_flow_max_level,
            max_disappeared=optical_flow_max_disappeared,
            min_tracked_points=optical_flow_min_tracked_points
        )
        
        # 最適化: メモリ事前確保
        self._result_cache: Dict[str, Any] = {}
    
    def predict_gdino(self,
                     images_pil: List[Image.Image],
                     texts_prompt: List[str],
                     box_threshold: float = 0.3,
                     text_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """GroundingDINO予測（最適化版）"""
        # 目的: 自然言語プロンプトによるゼロショット物体検出を実行する目的で使用
        return self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
    
    def track(self, 
             frame: np.ndarray,
             detections: Union[List, np.ndarray],
             labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """OpticalFlowトラッキング実行（特徴点版）"""
        # 目的: 検出結果に対してOpticalFlowトラッキングを実行し、堅牢な追跡を実現する目的で使用
        tracks, track_labels = self.optical_tracker.update(frame, detections, labels)
        
        result = {
            "boxes": tracks[:, :4].tolist() if len(tracks) > 0 else [],
            "labels": [track_labels.get(int(t[4]), "unknown") for t in tracks] if len(tracks) > 0 else [],
            "scores": [1.0] * len(tracks),
            "track_ids": tracks[:, 4].tolist() if len(tracks) > 0 else [],
            "masks": []
        }
        return result
    
    def visualize(self, 
                 image: np.ndarray,
                 result: Dict[str, Any]) -> np.ndarray:
        """可視化（最適化版）"""
        # 目的: 検出・追跡結果を画像上に可視化する目的で使用
        from .utils import draw_image
        
        # result辞書から必要な要素を抽出
        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        masks = result.get("masks", [])
        track_ids = result.get("track_ids", [])
        
        # numpy配列に変換
        if boxes:
            xyxy = np.array(boxes, dtype=np.float32)
            probs = np.array(scores, dtype=np.float32) if scores else np.ones(len(boxes), dtype=np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            probs = np.empty(0, dtype=np.float32)
            labels = []
        
        # マスクの処理
        if masks and len(masks) > 0:
            masks_array = np.array(masks)
        else:
            masks_array = None
        
        return draw_image(
            image_rgb=image,
            masks=masks_array,
            xyxy=xyxy,
            probs=probs,
            labels=labels,
            track_ids=track_ids
        )
    
    def clear_trackers(self) -> None:
        """トラッカークリア"""
        # 目的: トラッキング状態をリセットする目的で使用
        self.optical_tracker.reset()