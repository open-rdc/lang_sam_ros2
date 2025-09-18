import numpy as np
from PIL import Image

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam.models.utils import DEVICE


class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type

        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path, device=device)
        self.gdino = GDINO()
        self.gdino.build_model(device=device)

    def predict(
        self,
        images_pil: list[Image.Image],
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Predict detections and generate masks for given images and text prompt.
        
        Args:
            images_pil: List of PIL Images
            text_prompt: Text description of objects to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching confidence threshold
            
        Returns:
            List of dictionaries containing detection results for each image
        """
        results = []
        
        for image_pil in images_pil:
            # GroundingDINO detection
            detections = self.gdino.predict(
                image_pil, text_prompt, box_threshold, text_threshold
            )
            
            if len(detections) == 0:
                results.append({
                    "masks": np.array([]),
                    "boxes": np.array([]).reshape(0, 4),
                    "labels": [],
                    "scores": np.array([])
                })
                continue
                
            # Convert to format expected by SAM2
            boxes = []
            labels = []
            scores = []
            
            for detection in detections:
                x, y, w, h = detection.x, detection.y, detection.width, detection.height
                boxes.append([x, y, x + w, y + h])
                labels.append(detection.label)
                scores.append(detection.score)
            
            # SAM2 segmentation
            masks = self.sam.predict(image_pil, boxes)
            
            results.append({
                "masks": masks,
                "boxes": np.array(boxes),
                "labels": labels,
                "scores": np.array(scores)
            })
            
        return results