#!/usr/bin/env python3
"""
LangSAM Tracker - OpticalFlow版
GroundingDINO + SAM2 + OpticalFlow特徴点追跡統合システム
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple, Union
from functools import lru_cache

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
        
        # 最適化: 配列事前割り当て
        self._temp_tracks = np.empty((0, 5), dtype=np.float32)
        self._temp_labels = {}
        
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
        # 高速グレースケール変換
        # 目的: Optical Flow計算のためグレースケール化する目的で使用
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame  # 既にグレースケールの場合
        
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
            
            print(f"[OpticalFlowTracker] ROI({x1},{y1},{x2},{y2}) 特徴点検出: {len(corners) if corners is not None else 0}個")
            
            if corners is not None and len(corners) >= self.min_tracked_points:
                print(f"[OpticalFlowTracker] 変換前corners.shape: {corners.shape}")
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
                print(f"[OpticalFlowTracker] 新規トラック{track_id}を登録: {len(corners)}個の特徴点, corners.shape={corners.shape}")
    
    def _track_existing_objects(self, gray: np.ndarray):
        """既存トラックの特徴点を追跡（最適化版）"""
        # 目的: calcOpticalFlowPyrLKで前フレームから現フレームへの特徴点移動を計算する目的で使用
        
        # バッチ処理用の配列準備
        track_ids_to_process = []
        all_old_points = []
        
        # 有効な特徴点を持つトラックを収集
        for track_id, track in self.tracks.items():
            if track["points"] is not None and len(track["points"]) > 0:
                track_ids_to_process.append(track_id)
                all_old_points.extend(track["points"])
        
        print(f"[OpticalFlowTracker] 追跡対象: {len(track_ids_to_process)}個のトラック, {len(all_old_points)}個の特徴点")
        
        if not all_old_points:
            print("[OpticalFlowTracker] 追跡可能な特徴点がありません")
            # すべてのトラックのdisappearedをインクリメント
            for track in self.tracks.values():
                track["disappeared"] += 1
            return
        
        # バッチでOptical Flow計算（高速化）
        all_old_points_array = np.array(all_old_points, dtype=np.float32)
        print(f"[OpticalFlowTracker] Optical Flow計算開始: {all_old_points_array.shape}の特徴点")
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, all_old_points_array, None, **self.lk_params
        )
        
        if status is not None:
            successful_points = np.sum(status == 1)
            print(f"[OpticalFlowTracker] Optical Flow完了: {successful_points}/{len(status)}個の特徴点追跡成功")
        
        # 結果を各トラックに分配
        point_idx = 0
        for track_id in track_ids_to_process:
            track = self.tracks[track_id]
            track_points_count = len(track["points"])
            
            track_new_points = new_points[point_idx:point_idx + track_points_count]
            track_status = status[point_idx:point_idx + track_points_count]
            track_old_points = track["points"]
            
            point_idx += track_points_count
            
            # 追跡成功した特徴点のみを保持
            good_new = track_new_points[track_status.flatten() == 1]
            good_old = track_old_points[track_status.flatten() == 1]
            
            # 特徴点の形状と数をチェック
            if len(good_new) >= self.min_tracked_points and len(good_old) >= self.min_tracked_points and good_new.shape == good_old.shape:
                try:
                    # 特徴点群の移動ベクトルを計算（中央値使用で外れ値除去）
                    # 目的: 2次元座標として整形して堅牢な移動ベクトル計算を実現する目的で使用
                    if good_new.ndim == 3:
                        good_new_2d = good_new.reshape(-1, 2)
                        good_old_2d = good_old.reshape(-1, 2)
                    else:
                        good_new_2d = good_new
                        good_old_2d = good_old
                    
                    # 中央値を使用して外れ値に対する堅牢性を向上
                    differences = good_new_2d - good_old_2d
                    movement = np.median(differences, axis=0)
                    
                    # 移動ベクトル抽出（次元チェック付き）
                    if movement.ndim == 1 and len(movement) >= 2:
                        dx, dy = float(movement[0]), float(movement[1])
                    elif movement.ndim == 2 and movement.shape[0] >= 2:
                        dx, dy = float(movement[0]), float(movement[1])
                    else:
                        # フォールバック: 移動なしとして処理
                        dx, dy = 0.0, 0.0
                    
                    # 異常値のチェック（大きすぎる移動を制限）
                    max_movement = 100.0  # 最大移動ピクセル数
                    if abs(dx) > max_movement or abs(dy) > max_movement:
                        dx, dy = 0.0, 0.0
                    
                    # BBOXを更新
                    old_bbox = track["bbox"]
                    new_bbox = [
                        old_bbox[0] + dx,
                        old_bbox[1] + dy,
                        old_bbox[2] + dx,
                        old_bbox[3] + dy
                    ]
                    
                    # 画像境界内にクリップ（高速化）
                    h, w = gray.shape
                    new_bbox[0] = np.clip(new_bbox[0], 0, w-1)
                    new_bbox[1] = np.clip(new_bbox[1], 0, h-1)
                    new_bbox[2] = np.clip(new_bbox[2], new_bbox[0]+1, w)
                    new_bbox[3] = np.clip(new_bbox[3], new_bbox[1]+1, h)
                    
                    # トラック情報を更新
                    track["prev_points"] = track["points"]  # 前フレームの特徴点を保存
                    track["points"] = good_new.reshape(-1, 1, 2)
                    track["bbox"] = new_bbox
                    track["disappeared"] = 0
                    
                    print(f"[OpticalFlowTracker] トラック{track_id}更新: 移動(dx={dx:.1f}, dy={dy:.1f}), {len(good_new)}個の特徴点")
                    
                except Exception as e:
                    # エラー時は追跡失敗として扱う
                    print(f"[OpticalFlowTracker] トラック{track_id}移動計算エラー: {e}")
                    track["disappeared"] += 1
                    track["points"] = None
            else:
                # 追跡できた特徴点が少なすぎる
                track["disappeared"] += 1
                track["points"] = None
        
        # 残りのトラック（特徴点がないもの）のdisappearedをインクリメント
        for track_id, track in self.tracks.items():
            if track_id not in track_ids_to_process:
                track["disappeared"] += 1
    
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
        
        # 既存トラックのbbox一覧を事前に取得（高速化）
        active_tracks = {tid: track["bbox"] for tid, track in self.tracks.items() 
                        if track["bbox"] is not None}
        
        # バッチでIoU計算（ベクトル化）
        if active_tracks:
            track_ids = list(active_tracks.keys())
            track_bboxes = np.array([active_tracks[tid] for tid in track_ids], dtype=np.float32)
            
            for i, det in enumerate(dets):
                bbox = det[:4].astype(np.float32)
                
                # ベクトル化されたIoU計算
                ious = self._calculate_iou_vectorized(bbox, track_bboxes)
                
                # 最良のマッチを検索
                best_idx = np.argmax(ious)
                best_iou = ious[best_idx]
                
                if best_iou > 0.3:  # IoU閾値
                    best_track_id = track_ids[best_idx]
            
                    # 既存トラックを更新
                    self._update_track_features(gray, bbox.astype(int), best_track_id, 
                                               labels[i] if labels and i < len(labels) else "unknown")
                else:
                    # 新規トラック登録
                    self._create_new_track(gray, bbox.astype(int), 
                                         labels[i] if labels and i < len(labels) else "unknown")
        else:
            # 既存トラックがない場合、すべて新規登録
            for i, det in enumerate(dets):
                bbox = det[:4].astype(int)
                self._create_new_track(gray, bbox, 
                                     labels[i] if labels and i < len(labels) else "unknown")
    
    def _update_track_features(self, gray: np.ndarray, bbox: np.ndarray, track_id: int, label: str):
        """トラックの特徴点を更新"""
        x1, y1, x2, y2 = bbox
        if y2 <= y1 or x2 <= x1:  # 無効なbbox
            return
            
        roi = gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)
        
        if corners is not None and len(corners) >= self.min_tracked_points:
            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
            self.tracks[track_id]["points"] = corners
            self.tracks[track_id]["bbox"] = bbox.tolist()
            self.tracks[track_id]["disappeared"] = 0
    
    def _create_new_track(self, gray: np.ndarray, bbox: np.ndarray, label: str):
        """新規トラックを作成"""
        x1, y1, x2, y2 = bbox
        if y2 <= y1 or x2 <= x1:  # 無効なbbox
            return
            
        roi = gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)
        
        if corners is not None and len(corners) >= self.min_tracked_points:
            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
            
            track_id = self.next_id
            self.tracks[track_id] = {
                "points": corners,
                "bbox": bbox.tolist(),
                "label": label,
                "disappeared": 0
            }
            self.next_id += 1
    
    def _calculate_iou_vectorized(self, bbox1: np.ndarray, bbox2_array: np.ndarray) -> np.ndarray:
        """IoU（Intersection over Union）をベクトル化して高速計算"""
        # 目的: 複数のBBOXと1つのBBOXのIoUを一括計算する目的で使用
        if len(bbox2_array) == 0:
            return np.array([])
            
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2_array[:, 0], bbox2_array[:, 1], bbox2_array[:, 2], bbox2_array[:, 3]
        
        # ベクトル化された交差領域計算
        x_inter_min = np.maximum(x1_min, x2_min)
        y_inter_min = np.maximum(y1_min, y2_min)
        x_inter_max = np.minimum(x1_max, x2_max)
        y_inter_max = np.minimum(y1_max, y2_max)
        
        # 交差領域の幅と高さ
        inter_width = np.maximum(0, x_inter_max - x_inter_min)
        inter_height = np.maximum(0, y_inter_max - y_inter_min)
        inter_area = inter_width * inter_height
        
        # 各BBOXの面積
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_areas = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Union面積
        union_areas = bbox1_area + bbox2_areas - inter_area
        
        # IoU計算（0除算防止）
        return np.divide(inter_area, union_areas, out=np.zeros_like(inter_area), where=union_areas!=0)
    
    def _cleanup_tracks(self):
        """disappeared countが閾値を超えたトラックを削除（最適化版）"""
        # 目的: 長時間検出されないトラックを削除する目的で使用
        tracks_to_remove = [track_id for track_id, track in self.tracks.items() 
                           if track["disappeared"] > self.max_disappeared]
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _get_current_tracks(self) -> Tuple[np.ndarray, Dict[str, str]]:
        """現在のトラック情報を取得（最適化版）"""
        # 目的: 有効なトラック情報を整形して返す目的で使用
        if not self.tracks:
            return self._temp_tracks[:0], {}
        
        # 有効なトラックをフィルタリング
        active_tracks = [(track_id, track) for track_id, track in self.tracks.items() 
                        if track["bbox"] is not None and track["disappeared"] == 0]
        
        if not active_tracks:
            return self._temp_tracks[:0], {}
        
        # 効率的な配列構築
        num_tracks = len(active_tracks)
        if len(self._temp_tracks) < num_tracks:
            self._temp_tracks = np.empty((num_tracks * 2, 5), dtype=np.float32)  # バッファサイズを倍に
        
        tracks_array = self._temp_tracks[:num_tracks]
        self._temp_labels.clear()
        
        for i, (track_id, track) in enumerate(active_tracks):
            bbox = track["bbox"]
            tracks_array[i] = [bbox[0], bbox[1], bbox[2], bbox[3], float(track_id)]
            self._temp_labels[track_id] = track["label"]
        
        return tracks_array, self._temp_labels.copy()
    
    def visualize_optical_flow(self, image: np.ndarray, use_draw_image: bool = False) -> np.ndarray:
        """OpticalFlow可視化（ハイブリッド版：draw_image + OpenCV）"""
        # 目的: draw_imageでBBOX・ラベルを描画し、OpenCVで特徴点・移動ベクトルを追加描画する目的で使用
        
        if use_draw_image:
            # まずdraw_imageで基本要素（BBOX、ラベル、マスク）を描画
            from .utils import draw_image
            
            # トラッキング結果をdraw_image形式に変換
            boxes = []
            labels = []
            scores = []
            track_ids = []
            
            for track_id, track in self.tracks.items():
                if track["bbox"] is not None:
                    boxes.append(track["bbox"])
                    labels.append(track["label"])
                    scores.append(1.0)  # 固定スコア
                    track_ids.append(track_id)
            
            if boxes:
                xyxy = np.array(boxes, dtype=np.float32)
                probs = np.array(scores, dtype=np.float32)
                
                # draw_imageで基本描画（ID表示なし）
                vis_image = draw_image(
                    image_rgb=image,
                    masks=None,
                    xyxy=xyxy,
                    probs=probs,
                    labels=labels,
                    track_ids=[]  # ID表示を削除
                )
            else:
                vis_image = image.copy()
        else:
            vis_image = image.copy()
        
        # OpenCVでOpticalFlow専用要素を追加描画（赤色固定）
        red_color = (255, 0, 0)  # RGB形式での赤色（draw_imageと統一）
        for track_id, track in self.tracks.items():
            if track["bbox"] is None or track["points"] is None:
                continue
                
            points = track["points"]
            
            # 特徴点描画（小さな赤い円）
            if len(points) > 0:
                for point in points:
                    if point.shape == (1, 2):
                        x, y = point[0]
                        cv2.circle(vis_image, (int(x), int(y)), 2, red_color, -1)
                    elif point.shape == (2,):
                        x, y = point
                        cv2.circle(vis_image, (int(x), int(y)), 2, red_color, -1)
            
            # 前フレームの特徴点がある場合、移動ベクトル描画
            if hasattr(track, 'prev_points') and track.get('prev_points') is not None:
                prev_points = track['prev_points']
                curr_points = points
                
                if len(prev_points) == len(curr_points):
                    for prev_pt, curr_pt in zip(prev_points, curr_points):
                        if prev_pt.shape == (1, 2) and curr_pt.shape == (1, 2):
                            x1, y1 = prev_pt[0].astype(int)
                            x2, y2 = curr_pt[0].astype(int)
                            cv2.arrowedLine(vis_image, (x1, y1), (x2, y2), red_color, 1, tipLength=0.2)
        
        return vis_image
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """トラックID別の色を生成"""
        # 目的: トラック識別のための固定色を生成する目的で使用
        colors = [
            (0, 255, 0),    # 緑
            (255, 0, 0),    # 青
            (0, 0, 255),    # 赤
            (255, 255, 0),  # シアン
            (255, 0, 255),  # マゼンタ
            (0, 255, 255),  # 黄
            (128, 0, 128),  # 紫
            (255, 165, 0),  # オレンジ
        ]
        return colors[track_id % len(colors)]

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
    
    def process_frame(self,
                     frame: np.ndarray,
                     gdino_result: Optional[Dict[str, Any]] = None,
                     reset_tracker: bool = False) -> Dict[str, Any]:
        """統合フレーム処理（GroundingDINO結果とOpticalFlowトラッキング）"""
        # 目的: GroundingDINO検出結果をOpticalFlowトラッキング用に変換して処理する目的で使用
        
        # トラッカーリセット（必要時）
        if reset_tracker and gdino_result and len(gdino_result.get('boxes', [])) > 0:
            self.clear_trackers()
        
        # 検出結果の形式変換（高速化）
        if gdino_result and len(gdino_result.get('boxes', [])) > 0:
            # PyTorchテンソルをnumpy配列に変換
            # 目的: PyTorchテンソルをnumpy配列に変換してTensorのBoolean判定エラーを回避する目的で使用
            boxes_tensor = gdino_result.get('boxes', [])
            scores_tensor = gdino_result.get('scores', [])
            
            boxes_np = boxes_tensor.cpu().numpy() if hasattr(boxes_tensor, 'cpu') else np.array(boxes_tensor)
            scores_np = scores_tensor.cpu().numpy() if hasattr(scores_tensor, 'cpu') else np.array(scores_tensor)
            
            # ベクトル化されたdetections配列の作成 [x1, y1, x2, y2, score]
            if len(scores_np) > 0:
                detections = np.column_stack([boxes_np, scores_np[:len(boxes_np)]])
            else:
                detections = np.column_stack([boxes_np, np.ones(len(boxes_np))])
            
            labels = gdino_result.get("labels", [])
        else:
            detections = []
            labels = []
        
        # OpticalFlowトラッキング実行
        return self.track(frame, detections, labels)
    
    def track(self, 
             frame: np.ndarray,
             detections: Union[List, np.ndarray],
             labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """OpticalFlowトラッキング実行（特徴点版・高速化）"""
        # 目的: 検出結果に対してOpticalFlowトラッキングを実行し、堅牢な追跡を実現する目的で使用
        tracks, track_labels = self.optical_tracker.update(frame, detections, labels)
        
        # 高速化: 条件分岐を最小化
        if len(tracks) > 0:
            boxes = tracks[:, :4].tolist()
            track_ids_list = tracks[:, 4].tolist()
            labels_list = [track_labels.get(int(tid), "unknown") for tid in track_ids_list]
            scores = [1.0] * len(tracks)
        else:
            boxes = []
            track_ids_list = []
            labels_list = []
            scores = []
        
        return {
            "boxes": boxes,
            "labels": labels_list,
            "scores": scores,
            "track_ids": track_ids_list,
            "masks": []
        }
    
    def visualize(self, 
                 image: np.ndarray,
                 result: Dict[str, Any]) -> np.ndarray:
        """可視化（最適化版）"""
        # 目的: 検出・追跡結果を画像上に可視化する目的で使用
        from .utils import draw_image
        
        # result辞書から必要な要素を抽出（高速化）
        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        masks = result.get("masks", [])
        track_ids = result.get("track_ids", [])
        
        # 事前チェックで処理を最適化
        if not boxes:
            return draw_image(
                image_rgb=image,
                masks=None,
                xyxy=np.empty((0, 4), dtype=np.float32),
                probs=np.empty(0, dtype=np.float32),
                labels=[],
                track_ids=[]
            )
        
        # numpy配列に変換（高速化）
        xyxy = np.array(boxes, dtype=np.float32)
        probs = np.array(scores, dtype=np.float32) if scores else np.ones(len(boxes), dtype=np.float32)
        
        # マスクの処理（高速化）
        masks_array = np.array(masks) if masks and len(masks) > 0 else None
        
        return draw_image(
            image_rgb=image,
            masks=masks_array,
            xyxy=xyxy,
            probs=probs,
            labels=labels,
            track_ids=track_ids
        )
    
    def visualize_optical_flow(self, image: np.ndarray) -> np.ndarray:
        """OpticalFlow可視化（ハイブリッド版）"""
        # 目的: draw_imageでBBOX・ラベルを描画し、OpenCVで特徴点・移動ベクトルを追加する目的で使用
        return self.optical_tracker.visualize_optical_flow(image, use_draw_image=True)
    
    def clear_trackers(self) -> None:
        """トラッカークリア"""
        # 目的: トラッキング状態をリセットする目的で使用
        self.optical_tracker.reset()