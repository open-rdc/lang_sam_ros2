"""
特徴点プール方式: 特徴点管理の改善
特徴点の自動補充と品質管理を行う
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging
from dataclasses import dataclass
from enum import Enum

class FeatureQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class FeaturePoint:
    """特徴点の情報を保持するデータクラス"""
    point: np.ndarray  # [x, y]
    quality: float
    age: int  # 何フレーム追跡されているか
    last_seen: float  # 最後に見つかった時刻
    reliability: float  # 信頼度（0-1）
    
class FeaturePool:
    """特徴点プール管理クラス"""
    
    def __init__(self, max_features: int = 100, min_features: int = 20):
        self.max_features = max_features
        self.min_features = min_features
        self.logger = logging.getLogger(__name__)
        
        # 特徴点プール
        self.active_features: Dict[str, List[FeaturePoint]] = {}
        self.backup_features: Dict[str, List[FeaturePoint]] = {}
        
        # Harris corner検出パラメータ
        self.harris_params = {
            'max_corners': max_features,
            'quality_level': 0.01,
            'min_distance': 10,
            'block_size': 3,
            'harris_k': 0.04,
            'use_harris_detector': True
        }
        
        # 特徴点品質管理
        self.quality_thresholds = {
            FeatureQuality.HIGH: 0.8,
            FeatureQuality.MEDIUM: 0.5,
            FeatureQuality.LOW: 0.2
        }
        
        # 特徴点の寿命管理
        self.max_age = 30  # 最大30フレーム
        self.reliability_decay = 0.95  # 信頼度減衰率
        
    def initialize_features(self, label: str, gray_image: np.ndarray, bbox: List[float]) -> bool:
        """指定されたラベルの特徴点を初期化"""
        try:
            # バウンディングボックス内から特徴点を抽出
            features = self._extract_features_from_bbox(gray_image, bbox)
            
            if features is None or len(features) == 0:
                self.logger.warning(f"ラベル {label} の特徴点抽出に失敗")
                return False
            
            # FeaturePointオブジェクトを作成
            feature_points = []
            for feature in features:
                fp = FeaturePoint(
                    point=feature.flatten(),
                    quality=1.0,  # 初期品質は最高
                    age=0,
                    last_seen=0.0,
                    reliability=1.0
                )
                feature_points.append(fp)
            
            self.active_features[label] = feature_points
            self.backup_features[label] = []
            
            self.logger.info(f"ラベル {label} で {len(feature_points)} 個の特徴点を初期化")
            return True
            
        except Exception as e:
            self.logger.error(f"特徴点初期化エラー: {e}")
            return False
    
    def update_features(self, label: str, new_features: np.ndarray, status: np.ndarray, 
                       current_time: float) -> bool:
        """特徴点を更新"""
        try:
            if label not in self.active_features:
                return False
            
            active_fps = self.active_features[label]
            if len(active_fps) != len(new_features):
                self.logger.warning(f"特徴点数が一致しません: {len(active_fps)} vs {len(new_features)}")
                return False
            
            # 有効な特徴点のみを更新
            updated_features = []
            for i, (fp, new_point, is_valid) in enumerate(zip(active_fps, new_features, status)):
                if is_valid:
                    # 特徴点を更新
                    fp.point = new_point.flatten()
                    fp.age += 1
                    fp.last_seen = current_time
                    fp.reliability *= self.reliability_decay
                    
                    # 年齢チェック
                    if fp.age < self.max_age:
                        updated_features.append(fp)
                    else:
                        # 古い特徴点をバックアップに移動
                        self._move_to_backup(label, fp)
                else:
                    # 無効な特徴点もバックアップに移動
                    self._move_to_backup(label, fp)
            
            self.active_features[label] = updated_features
            
            # 特徴点が不足している場合は補充
            if len(updated_features) < self.min_features:
                self.logger.info(f"ラベル {label} の特徴点が不足: {len(updated_features)}/{self.min_features}")
                return False  # 特徴点補充が必要
            
            return True
            
        except Exception as e:
            self.logger.error(f"特徴点更新エラー: {e}")
            return False
    
    def refresh_features(self, label: str, gray_image: np.ndarray, bbox: List[float]) -> bool:
        """特徴点を補充"""
        try:
            if label not in self.active_features:
                return self.initialize_features(label, gray_image, bbox)
            
            current_count = len(self.active_features[label])
            needed_count = self.min_features - current_count
            
            if needed_count <= 0:
                return True
            
            # 新しい特徴点を抽出
            new_features = self._extract_features_from_bbox(
                gray_image, bbox, max_corners=needed_count
            )
            
            if new_features is None or len(new_features) == 0:
                self.logger.warning(f"ラベル {label} の特徴点補充に失敗")
                return False
            
            # 既存の特徴点との重複を避ける
            existing_points = np.array([fp.point for fp in self.active_features[label]])
            filtered_features = self._filter_duplicate_features(new_features, existing_points)
            
            # 新しい特徴点を追加
            for feature in filtered_features:
                fp = FeaturePoint(
                    point=feature.flatten(),
                    quality=0.8,  # 新しい特徴点は中程度の品質
                    age=0,
                    last_seen=0.0,
                    reliability=0.8
                )
                self.active_features[label].append(fp)
            
            self.logger.info(f"ラベル {label} で {len(filtered_features)} 個の特徴点を補充")
            return True
            
        except Exception as e:
            self.logger.error(f"特徴点補充エラー: {e}")
            return False
    
    def get_active_features(self, label: str) -> Optional[np.ndarray]:
        """アクティブな特徴点を取得"""
        if label not in self.active_features:
            return None
        
        if len(self.active_features[label]) == 0:
            return None
        
        features = np.array([fp.point for fp in self.active_features[label]])
        return features.reshape(-1, 1, 2).astype(np.float32)
    
    def get_feature_quality(self, label: str) -> FeatureQuality:
        """特徴点の品質を評価"""
        if label not in self.active_features:
            return FeatureQuality.LOW
        
        features = self.active_features[label]
        if len(features) == 0:
            return FeatureQuality.LOW
        
        # 平均信頼度を計算
        avg_reliability = np.mean([fp.reliability for fp in features])
        
        if avg_reliability >= self.quality_thresholds[FeatureQuality.HIGH]:
            return FeatureQuality.HIGH
        elif avg_reliability >= self.quality_thresholds[FeatureQuality.MEDIUM]:
            return FeatureQuality.MEDIUM
        else:
            return FeatureQuality.LOW
    
    def remove_label(self, label: str) -> None:
        """ラベルを削除"""
        if label in self.active_features:
            del self.active_features[label]
        if label in self.backup_features:
            del self.backup_features[label]
    
    def _extract_features_from_bbox(self, gray_image: np.ndarray, bbox: List[float], 
                                   max_corners: Optional[int] = None) -> Optional[np.ndarray]:
        """バウンディングボックス内から特徴点を抽出"""
        try:
            if len(bbox) != 4:
                return None
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            h, w = gray_image.shape[:2]
            
            # 境界チェック
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # ROI抽出
            roi = gray_image[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            
            # Harris corner検出
            corners = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=max_corners or self.harris_params['max_corners'],
                qualityLevel=self.harris_params['quality_level'],
                minDistance=self.harris_params['min_distance'],
                blockSize=self.harris_params['block_size'],
                k=self.harris_params['harris_k'],
                useHarrisDetector=self.harris_params['use_harris_detector']
            )
            
            if corners is None:
                return None
            
            # 絶対座標に変換
            corners[:, :, 0] += x1
            corners[:, :, 1] += y1
            
            return corners
            
        except Exception as e:
            self.logger.error(f"特徴点抽出エラー: {e}")
            return None
    
    def _filter_duplicate_features(self, new_features: np.ndarray, 
                                 existing_points: np.ndarray, 
                                 min_distance: float = 10.0) -> np.ndarray:
        """重複する特徴点をフィルタリング"""
        if len(existing_points) == 0:
            return new_features
        
        filtered = []
        for new_feature in new_features:
            new_point = new_feature.flatten()
            
            # 既存の特徴点との距離をチェック
            distances = np.linalg.norm(existing_points - new_point, axis=1)
            
            if np.min(distances) >= min_distance:
                filtered.append(new_feature)
        
        return np.array(filtered) if filtered else np.array([]).reshape(0, 1, 2)
    
    def _move_to_backup(self, label: str, feature_point: FeaturePoint) -> None:
        """特徴点をバックアップに移動"""
        if label not in self.backup_features:
            self.backup_features[label] = []
        
        # バックアップの容量制限
        if len(self.backup_features[label]) >= self.max_features // 2:
            self.backup_features[label].pop(0)  # 古いものを削除
        
        self.backup_features[label].append(feature_point)
    
    def get_statistics(self) -> Dict[str, Dict]:
        """統計情報を取得"""
        stats = {}
        
        for label in self.active_features:
            active_count = len(self.active_features[label])
            backup_count = len(self.backup_features.get(label, []))
            quality = self.get_feature_quality(label)
            
            if active_count > 0:
                avg_age = np.mean([fp.age for fp in self.active_features[label]])
                avg_reliability = np.mean([fp.reliability for fp in self.active_features[label]])
            else:
                avg_age = 0
                avg_reliability = 0
            
            stats[label] = {
                'active_count': active_count,
                'backup_count': backup_count,
                'quality': quality.value,
                'avg_age': avg_age,
                'avg_reliability': avg_reliability
            }
        
        return stats