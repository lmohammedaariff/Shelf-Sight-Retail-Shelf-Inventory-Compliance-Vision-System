"""End-to-end region proposals and crop-level SKU inference."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np

from .classification import SKUClassifier, classify_crop
from .config import VisionConfig
from .region_proposals import Box, propose_regions

logger = logging.getLogger(__name__)


def validate_manual_boxes(boxes: object, image_width: int, image_height: int,
                          *, max_regions: int = 100) -> list[Box]:
    """Validate and clip normalized or pixel xywh rectangles to image bounds."""
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("Manual regions must be a non-empty JSON list of [x, y, width, height] boxes.")
    if len(boxes) > max_regions:
        raise ValueError(f"At most {max_regions} manual regions are allowed.")
    converted: list[Box] = []
    for index, row in enumerate(boxes):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"Region {index + 1} must contain exactly four numeric values.")
        try:
            x, y, width, height = (float(value) for value in row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Region {index + 1} contains a non-numeric value.") from exc
        values = (x, y, width, height)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Region {index + 1} contains a non-finite value.")
        if width <= 0 or height <= 0:
            raise ValueError(f"Region {index + 1} must have positive width and height.")
        if all(0 <= value <= 1 for value in values):
            x, width = x * image_width, width * image_width
            y, height = y * image_height, height * image_height
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            raise ValueError(f"Region {index + 1} falls outside the image bounds.")
        converted.append((round(x), round(y), round(width), round(height)))
    return converted


def analyze_shelf(rgb: np.ndarray, classifier: SKUClassifier | None = None, *, manual_boxes=None,
                  sensitivity: int = 5, threshold: float | None = None,
                  config: VisionConfig | None = None, image_id: str = "uploaded-image") -> dict:
    """Detect candidate shelf products, classify crops, and render annotations."""
    config = config or VisionConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError("rgb must be a non-empty H×W×3 uint8 image")
    height, width = rgb.shape[:2]
    if manual_boxes is not None:
        boxes = validate_manual_boxes(manual_boxes, width, height, max_regions=config.max_regions)
        proposal_method = "manual"
    else:
        boxes = propose_regions(rgb, sensitivity, resize_max_side=config.resize_max_side,
                                max_regions=config.max_regions,
                                nms_iou_threshold=config.nms_iou_threshold)
        proposal_method = "opencv_classical_baseline"
    confidence_threshold = config.confidence_threshold if threshold is None else threshold
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    annotated = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    products: list[dict] = []
    for identifier, (x, y, box_width, box_height) in enumerate(boxes, start=1):
        crop = rgb[y:y + box_height, x:x + box_width]
        prediction = classify_crop(crop, classifier, confidence_threshold)
        product = {
            "id": identifier, "sku": prediction.sku,
            "confidence": round(prediction.confidence, 6) if prediction.confidence is not None else None,
            "status": prediction.status, "proposal_method": proposal_method,
            "image": image_id,
            "region_confidence": None, "x": x, "y": y, "width": box_width, "height": box_height,
            "image_width": width, "image_height": height,
        }
        products.append(product)
        color = (50, 190, 70) if prediction.status == "ACCEPTED" else (30, 165, 240)
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), color, max(2, width // 600))
        label = (f"{prediction.sku} {prediction.confidence:.0%}" if classifier and prediction.confidence is not None
                 else ("Model required" if prediction.status == "MODEL_REQUIRED" else "Unknown · review")
                 if len(boxes) <= 12 else "")
        if label:
            cv2.putText(annotated, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.45, width / 1600), color, 2, cv2.LINE_AA)
    return {"annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "products": products,
            "processing": {"proposal_method": proposal_method, "candidate_regions": len(boxes),
                           "classifier_loaded": classifier is not None}}
