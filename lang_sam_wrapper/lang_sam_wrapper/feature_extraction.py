"""
特徴点抽出とオプティカルフロー用のユーティリティ関数
"""

import cv2
import numpy as np
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
        print(f"Harris corner検出開始: bbox={bbox}")
        
        # バウンディングボックスの座標を整数に変換
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 画像境界内にクリップ
        height, width = gray_image.shape
        original_coords = (x1, y1, x2, y2)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        print(f"座標変換: 元={original_coords}, 調整後=({x1}, {y1}, {x2}, {y2}), 画像サイズ=({width}, {height})")
        
        # バウンディングボックス内の領域を抽出
        roi = gray_image[y1:y2, x1:x2]
        print(f"ROIサイズ: {roi.shape}")
        
        # ROIが小さすぎる場合は中心点を返す
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"ROIが小さすぎます。中心点を返します: ({center_x}, {center_y})")
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
        
        print(f"goodFeaturesToTrack結果: {corners is not None and len(corners) if corners is not None else 0}個の特徴点")
        
        # 検出された特徴点を元の画像座標系に変換
        if corners is not None and len(corners) > 0:
            corners[:, 0, 0] += x1  # x座標にオフセット追加
            corners[:, 0, 1] += y1  # y座標にオフセット追加
            print(f"特徴点座標変換完了: {len(corners)}個")
            return corners
        else:
            # 特徴点が見つからない場合は中心点を返す
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"特徴点が見つかりません。中心点を返します: ({center_x}, {center_y})")
            return np.array([[[center_x, center_y]]], dtype=np.float32)
            
    except Exception as e:
        print(f"Harris corner抽出エラー: {e}")
        import traceback
        print(f"トレースバック: {traceback.format_exc()}")
        # エラー時は中心点を返す
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return np.array([[[center_x, center_y]]], dtype=np.float32)


def select_best_feature_point(
    corners: np.ndarray,
    gray_image: np.ndarray,
    selection_method: str = "harris_response"
) -> np.ndarray:
    """
    複数の特徴点から最も信頼できる一点を選択
    
    Args:
        corners: 特徴点配列 shape: (N, 1, 2)
        gray_image: グレースケール画像
        selection_method: 選択方法 ("harris_response", "center_closest", "random")
        
    Returns:
        選択された特徴点 shape: (1, 1, 2)
    """
    if corners is None or len(corners) == 0:
        return None
    
    if len(corners) == 1:
        return corners
    
    try:
        if selection_method == "harris_response":
            # Harris応答値が最も高い点を選択
            best_idx = _select_by_harris_response(corners, gray_image)
        elif selection_method == "center_closest":
            # 中心に最も近い点を選択
            best_idx = _select_by_center_distance(corners, gray_image.shape)
        elif selection_method == "random":
            # ランダムに選択
            best_idx = np.random.randint(0, len(corners))
        else:
            # デフォルト: 最初の点を選択
            best_idx = 0
            
        return corners[best_idx:best_idx+1]
        
    except Exception as e:
        print(f"特徴点選択エラー: {e}")
        # エラー時は最初の点を返す
        return corners[0:1]


def _select_by_harris_response(
    corners: np.ndarray, 
    gray_image: np.ndarray,
    block_size: int = 3,
    harris_k: float = 0.04
) -> int:
    """
    Harris応答値に基づいて最良の特徴点を選択
    
    Args:
        corners: 特徴点配列
        gray_image: グレースケール画像
        block_size: Harris検出のブロックサイズ
        harris_k: Harris検出のkパラメータ
        
    Returns:
        最良の特徴点のインデックス
    """
    try:
        # Harris応答値を計算
        harris_response = cv2.cornerHarris(
            gray_image, 
            blockSize=block_size, 
            ksize=3, 
            k=harris_k
        )
        
        best_response = -1
        best_idx = 0
        
        # 各特徴点でのHarris応答値を評価
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            x_int, y_int = int(x), int(y)
            
            # 画像境界内かチェック
            if 0 <= x_int < harris_response.shape[1] and 0 <= y_int < harris_response.shape[0]:
                response = harris_response[y_int, x_int]
                if response > best_response:
                    best_response = response
                    best_idx = i
        
        return best_idx
        
    except Exception as e:
        print(f"Harris応答値計算エラー: {e}")
        return 0


def _select_by_center_distance(corners: np.ndarray, image_shape: Tuple[int, int]) -> int:
    """
    画像中心に最も近い特徴点を選択
    
    Args:
        corners: 特徴点配列
        image_shape: 画像の形状 (height, width)
        
    Returns:
        最良の特徴点のインデックス
    """
    center_x = image_shape[1] / 2
    center_y = image_shape[0] / 2
    
    best_distance = float('inf')
    best_idx = 0
    
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if distance < best_distance:
            best_distance = distance
            best_idx = i
    
    return best_idx


def extract_edge_centroid_from_bbox(
    gray_image: np.ndarray,
    bbox: List[float],
    canny_low_threshold: int = 50,
    canny_high_threshold: int = 150,
    min_edge_pixels: int = 10
) -> Optional[np.ndarray]:
    """
    バウンディングボックス内でエッジ検出を行い、エッジピクセルの重心を計算
    
    Args:
        gray_image: グレースケール画像
        bbox: バウンディングボックス [x1, y1, x2, y2]
        canny_low_threshold: Canny edge検出の低閾値
        canny_high_threshold: Canny edge検出の高閾値
        min_edge_pixels: 有効とみなす最小エッジピクセル数
        
    Returns:
        エッジの重心点 shape: (1, 1, 2) またはNone
    """
    try:
        print(f"エッジ検出開始: bbox={bbox}")
        
        # バウンディングボックスの座標を整数に変換
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 画像境界内にクリップ
        height, width = gray_image.shape
        original_coords = (x1, y1, x2, y2)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        print(f"座標変換: 元={original_coords}, 調整後=({x1}, {y1}, {x2}, {y2})")
        
        # バウンディングボックス内の領域を抽出
        roi = gray_image[y1:y2, x1:x2]
        print(f"ROIサイズ: {roi.shape}")
        
        # ROIが小さすぎる場合は中心点を返す
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"ROIが小さすぎます。中心点を返します: ({center_x}, {center_y})")
            return np.array([[[center_x, center_y]]], dtype=np.float32)
        
        # Canny edge検出
        edges = cv2.Canny(roi, canny_low_threshold, canny_high_threshold)
        
        # エッジピクセルの座標を取得
        edge_coords = np.where(edges > 0)
        edge_count = len(edge_coords[0])
        
        print(f"検出されたエッジピクセル数: {edge_count}")
        
        # エッジが少なすぎる場合は中心点を返す
        if edge_count < min_edge_pixels:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"エッジピクセルが不足。中心点を返します: ({center_x}, {center_y})")
            return np.array([[[center_x, center_y]]], dtype=np.float32)
        
        # エッジピクセルの重心を計算（ROI座標系）
        centroid_y = np.mean(edge_coords[0])
        centroid_x = np.mean(edge_coords[1])
        
        # 元の画像座標系に変換
        global_centroid_x = centroid_x + x1
        global_centroid_y = centroid_y + y1
        
        print(f"エッジ重心計算完了: ({global_centroid_x:.1f}, {global_centroid_y:.1f})")
        
        return np.array([[[global_centroid_x, global_centroid_y]]], dtype=np.float32)
        
    except Exception as e:
        print(f"エッジ重心計算エラー: {e}")
        import traceback
        print(f"トレースバック: {traceback.format_exc()}")
        # エラー時は中心点を返す
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return np.array([[[center_x, center_y]]], dtype=np.float32)


def validate_tracking_points(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    status: np.ndarray,
    max_displacement: float = 50.0,
    min_valid_ratio: float = 0.5
) -> Tuple[bool, np.ndarray]:
    """
    オプティカルフローの追跡結果を検証
    
    Args:
        prev_pts: 前フレームの特徴点
        curr_pts: 現フレームの特徴点
        status: 追跡ステータス
        max_displacement: 最大変位閾値
        min_valid_ratio: 有効な点の最小割合
        
    Returns:
        (追跡成功フラグ, 有効な特徴点)
    """
    try:
        if prev_pts is None or curr_pts is None or status is None:
            return False, None
        
        # 有効な点のみを抽出
        valid_indices = (status == 1).flatten()
        
        if not np.any(valid_indices):
            return False, None
        
        valid_prev = prev_pts[valid_indices]
        valid_curr = curr_pts[valid_indices]
        
        # 変位を計算
        displacement = np.linalg.norm(valid_curr - valid_prev, axis=2)
        
        # 妥当な変位の点のみを保持
        reasonable_indices = displacement.flatten() < max_displacement
        
        if not np.any(reasonable_indices):
            return False, None
        
        final_points = valid_curr[reasonable_indices]
        
        # 有効な点の割合をチェック
        valid_ratio = len(final_points) / len(prev_pts)
        
        if valid_ratio < min_valid_ratio:
            return False, final_points
        
        return True, final_points
        
    except Exception as e:
        print(f"追跡点検証エラー: {e}")
        return False, None