import cv2
import numpy as np
import supervision as sv
from PIL import Image

# ユーティリティ関数群: LangSAMシステム全体で使用される汎用ヘルパー関数
# 検出結果の可視化、画像処理、データ変換などを統一インターフェースで提供

MIN_AREA = 100  # コンター面積の最小闾値（ノイズ除去用）

# デバッグモード制御：本番環境では False に設定することでパフォーマンスを最適化
DEBUG_LABELS = False


def load_image(image_path: str):
    """画像ファイルの読み込みとRGB変換
    
    PILを使用して画像ファイルをロードし、AIモデル入力用に
    RGB形式に正規化する目的で使用。RGBAやグレースケールをRGBに変換。
    
    Args:
        image_path: 読み込む画像ファイルのパス
    
    Returns:
        PIL.Image: RGB形式のPIL画像オブジェクト
    """
    return Image.open(image_path).convert("RGB")


def draw_image(image_rgb, masks, xyxy, probs, labels):
    """検出結果を画像上に可視化描画（ラベル整合性保証付き）
    
    LangSAMシステム全体で使用される標準可視化関数。
    Supervisionライブラリを使用して一貫したスタイルで描画する目的で使用。
    
    Args:
        image_rgb: RGB形式の入力画像 (NumPy配列)
        masks: セグメンテーションマスク配列 (None可)
        xyxy: バウンディングボックス座標 [x1, y1, x2, y2]形式
        probs: 信頼度スコア配列
        labels: オブジェクトラベル文字列配列
    
    Returns:
        np.ndarray: 描画結果が合成された画像
    
    技術的特徴：
    - データサイズ不整合の自動修正機能
    - class_idマッピングでラベルの一意性保証
    - エラーハンドリングでシステム安定性を確保
    """
    
    # デバッグ: 入力データの整合性チェック
    # パフォーマンスインパクト最小化のためデバッグモードでのみ有効化
    if DEBUG_LABELS:
        print(f"[draw_image] Input data: boxes={len(xyxy)}, labels={labels}, probs={len(probs)}")
    
    # 入力データの整合性チェックと自動修正
    # システム全体の安定性を維持するためのディフェンシブプログラミング
    if len(xyxy) != len(labels) or len(xyxy) != len(probs):
        if DEBUG_LABELS:
            print(f"[draw_image] Warning: Data size mismatch - boxes:{len(xyxy)}, labels:{len(labels)}, probs:{len(probs)}")
        # 最小サイズに合わせてデータトリミング：エラー回避と一貫性保証の目的で使用
        min_size = min(len(xyxy), len(labels), len(probs))
        xyxy = xyxy[:min_size]
        labels = labels[:min_size]
        probs = probs[:min_size]
        # マスクデータも同様にトリミング
        if hasattr(masks, '__len__') and len(masks) > min_size:
            masks = masks[:min_size]
    
    if len(labels) == 0:
        return image_rgb
    
    # Supervisionライブラリのアノテーター初期化
    # 一貫したスタイルと高品質な可視化を実現する目的で使用
    box_annotator = sv.BoxCornerAnnotator()   # バウンディングボックス描画（コーナーマーク付き）
    label_annotator = sv.LabelAnnotator()     # ラベルテキスト描画（背景付き）
    mask_annotator = sv.MaskAnnotator()       # セグメンテーションマスク描画（透明度合成）
    
    # ユニークラベルごとのclass_id作成（順序保持）
    # Supervisionライブラリが要求する数値IDへのマッピングを作成する目的で使用
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]
    
    # Debug: Label mapping verification
    if DEBUG_LABELS:
        for i, (label, cls_id) in enumerate(zip(labels, class_id)):
            print(f"[draw_image] Object[{i}]: label='{label}', class_id={cls_id}")

    # マスクデータの安全な処理
    # SAM2の出力形式の多様性に対応したロバストなデータ変換を行う目的で使用
    mask_data = None
    if hasattr(masks, '__len__') and len(masks) > 0:
        try:
            if isinstance(masks, np.ndarray):
                # NumPy配列形式のマスクをブール型に正規化
                mask_data = masks.astype(bool)
            else:
                # リスト形式のマスクをNumPy配列に変換後ブール型に正規化
                # SAM2が生成する様々なマスク形式に対応する目的で使用
                mask_data = np.array(masks).astype(bool) if masks else None
        except Exception as e:
            if DEBUG_LABELS:
                print(f"[draw_image] Mask processing error: {e}")
            mask_data = None

    # Supervision Detectionsオブジェクトの作成
    # 各アノテーターが必要とするデータを統一フォーマットで格納する目的で使用
    detections = sv.Detections(
        xyxy=np.array(xyxy),       # バウンディングボックス座標
        mask=mask_data,            # セグメンテーションマスク
        confidence=np.array(probs), # 信頼度スコア
        class_id=np.array(class_id), # クラスID（ラベルからマッピング）
    )
    
    # ステップバイステップ描画処理（エラー特定のため）
    # 各描画ステップでエラーが発生してもシステム全体が停止しないようにする目的で使用
    try:
        annotated_image = image_rgb.copy()  # 元画像のコピーで安全な編集を実現
        # 1. バウンディングボックス描画
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        # 2. ラベルテキスト描画
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        # 3. マスク描画（データがある場合のみ）
        if mask_data is not None:
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    except Exception as e:
        if DEBUG_LABELS:
            print(f"[draw_image] 描画エラー: {e}")
        return image_rgb
        
    return annotated_image


def get_contours(mask):
    """セグメンテーションマスクからコンターを抽出
    
    SAM2で生成されたブール型マスクからOpenCVで処理可能なコンターを抽出し、
    物体境界の幾何情報を取得する目的で使用。ナビゲーション系で利用。
    
    Args:
        mask: SAM2が生成したブール型マスク配列
    
    Returns:
        list: 有効なコンターのリスト（MIN_AREA以上の面積を持つもののみ）
    """
    # 次元正規化：3DマスクをOpenCV用の2D形式に変換
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    
    # データ型変換：ブール型からOpenCV用のunit8型に変換
    mask = mask.astype(np.uint8)
    mask *= 255  # 0-1範囲を0-255範囲にスケーリング
    
    # コンター抽出：外側境界のみを取得して処理速度を最適化する目的で使用
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 有効コンターのフィルタリング：ノイズ除去と処理効率化の目的で使用
    effContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
    return effContours


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_labelme_json(binary_masks, labels, image_size, image_path=None):
    """Generate a LabelMe format JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the LabelMe JSON file.
    """
    num_masks = binary_masks.shape[0]

    json_dict = {
        "version": "4.5.6",
        "imageHeight": image_size[0],
        "imageWidth": image_size[1],
        "imagePath": image_path,
        "flags": {},
        "shapes": [],
        "imageData": None,
    }

    # Convert to numpy if tensor
    if hasattr(binary_masks, 'numpy'):
        binary_masks = binary_masks.numpy()

    # Loop through the masks and add them to the JSON dictionary
    for i in range(num_masks):
        mask = binary_masks[i]
        label = labels[i]
        effContours = get_contours(mask)

        for effContour in effContours:
            points = contour_to_points(effContour)
            shape_dict = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points,
                "shape_type": "polygon",
            }

            json_dict["shapes"].append(shape_dict)

    return json_dict
