"""
特徴点抽出とオプティカルフロー用のユーティリティ関数
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union


def extract_harris_corners_from_bbox(
    gray_image: np.ndarray,
    bbox: List[float],
    max_corners: int = 10,
    quality_level: float = 0.01,
    min_distance: int = 10,
    block_size: int = 3,
    harris_k: float = 0.04,
    use_harris_detector: bool = True
) -> Optional[np.ndarray]:
    """
    バウンディングボックス内からHarris corner検出で特徴点を抽出
    
    Args:
        gray_image: グレースケール画像
        bbox: バウンディングボックス [x1, y1, x2, y2]
        max_corners: 最大特徴点数
        quality_level: 品質レベル（0.01推奨）
        min_distance: 特徴点間の最小距離
        block_size: コーナー検出のブロックサイズ
        harris_k: Harris検出器のkパラメータ
        use_harris_detector: Harris検出器を使用するか（False=Shi-Tomasi）
        
    Returns:
        特徴点配列 shape: (N, 1, 2) またはNone
    """
    try:
        
        # バウンディングボックスの座標を整数に変換
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 画像境界内にクリップ
        height, width = gray_image.shape
        original_coords = (x1, y1, x2, y2)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        
        # バウンディングボックス内の領域を抽出
        roi = gray_image[y1:y2, x1:x2]
        
        # ROIが小さすぎる場合は中心点を返す
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return np.array([[[center_x, center_y]]], dtype=np.float32)
        
        # Harris corner検出
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
            useHarrisDetector=use_harris_detector,
            k=harris_k
        )
        
        
        # 検出された特徴点を元の画像座標系に変換
        if corners is not None and len(corners) > 0:
            corners[:, 0, 0] += x1  # x座標にオフセット追加
            corners[:, 0, 1] += y1  # y座標にオフセット追加
            return corners
        else:
            # 特徴点が見つからない場合は中心点を返す
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return np.array([[[center_x, center_y]]], dtype=np.float32)
            
    except Exception as e:
        import traceback
        logging.getLogger(__name__).error(f"Harris corner抽出エラー: {e}\nトレースバック: {traceback.format_exc()}")
        # エラー時は中心点を返す
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return np.array([[[center_x, center_y]]], dtype=np.float32)


