import cv2
import numpy as np
import supervision as sv
from PIL import Image

MIN_AREA = 100

# Debug mode control (set to False in production)
DEBUG_LABELS = False


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image_rgb, masks, xyxy, probs, labels):
    """Draw detection results on image with label consistency guarantee"""
    
    # Debug: Input data consistency check
    if DEBUG_LABELS:
        print(f"[draw_image] Input data: boxes={len(xyxy)}, labels={labels}, probs={len(probs)}")
    
    # 入力データの整合性チェック
    if len(xyxy) != len(labels) or len(xyxy) != len(probs):
        if DEBUG_LABELS:
            print(f"[draw_image] Warning: Data size mismatch - boxes:{len(xyxy)}, labels:{len(labels)}, probs:{len(probs)}")
        # 最小サイズに合わせて調整
        min_size = min(len(xyxy), len(labels), len(probs))
        xyxy = xyxy[:min_size]
        labels = labels[:min_size]
        probs = probs[:min_size]
        if hasattr(masks, '__len__') and len(masks) > min_size:
            masks = masks[:min_size]
    
    if len(labels) == 0:
        return image_rgb
    
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    # Create class_id for each unique label (preserve order)
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
    mask_data = None
    if hasattr(masks, '__len__') and len(masks) > 0:
        try:
            if isinstance(masks, np.ndarray):
                mask_data = masks.astype(bool)
            else:
                # リスト形式のマスクを処理
                mask_data = np.array(masks).astype(bool) if masks else None
        except Exception as e:
            if DEBUG_LABELS:
                print(f"[draw_image] Mask processing error: {e}")
            mask_data = None

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        mask=mask_data,
        confidence=np.array(probs),
        class_id=np.array(class_id),
    )
    
    # Draw step by step (for error localization)
    try:
        annotated_image = image_rgb.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        if mask_data is not None:
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    except Exception as e:
        if DEBUG_LABELS:
            print(f"[draw_image] 描画エラー: {e}")
        return image_rgb
        
    return annotated_image


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
