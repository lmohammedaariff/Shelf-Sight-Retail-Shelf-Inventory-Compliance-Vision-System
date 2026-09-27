"""Optional RetinaNet model loading and inference through KerasHub."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RetinaNetShelfDetector:
    """Fine-tuned KerasHub detector plus its ordered SKU class mapping."""

    model: object
    labels: list[str]
    model_path: Path
    image_size: int = 800


def load_detector(model_path: Path, labels_path: Path, info_path: Path | None = None) -> RetinaNetShelfDetector | None:
    """Load an optional fine-tuned KerasHub RetinaNet model and validate labels."""
    if not model_path.exists() and not labels_path.exists():
        return None
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Both the RetinaNet model and detector_labels.json must exist.")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        import keras_hub  # noqa: F401 - registers serialized KerasHub model layers
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install requirements-detector.txt to load the TensorFlow detector.") from exc
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("detector_labels.json must be a non-empty list of SKU names.")
    model = tf.keras.models.load_model(model_path)
    info_path = info_path or model_path.with_name("detector_info.json")
    image_size = 800
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        image_size = int(info.get("image_size", image_size))
    return RetinaNetShelfDetector(model, labels, model_path, image_size)


def _letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize without changing aspect ratio and pad to a square detector input."""
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x, pad_y = (size - new_width) // 2, (size - new_height) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def detect_shelf(rgb: np.ndarray, detector: RetinaNetShelfDetector, threshold: float = 0.4,
                 image_id: str = "uploaded-image") -> dict:
    """Run full-image RetinaNet prediction and return review-aware, original-image boxes."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError("rgb must be a non-empty H×W×3 uint8 image")
    height, width = rgb.shape[:2]
    image_size = detector.image_size
    input_image, scale, pad_x, pad_y = _letterbox(rgb, image_size)
    # Training scales uint8 pixels to [0, 1] in the dataset pipeline; keep inference identical.
    output = detector.model.predict(input_image[None].astype(np.float32) / 255.0, verbose=0)
    boxes = np.asarray(output.get("boxes"))[0]
    class_ids = np.asarray(output.get("classes", output.get("labels")))[0]
    scores = np.asarray(output.get("confidence", output.get("scores")))[0]
    count = int(np.asarray(output.get("num_detections", [len(scores)])).reshape(-1)[0])
    products = []
    annotated = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    for box, class_id, score in zip(boxes[:count], class_ids[:count], scores[:count]):
        score, class_id = float(score), int(class_id)
        if score < threshold or not 0 <= class_id < len(detector.labels):
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = (x1 - pad_x) / scale, (x2 - pad_x) / scale
        y1, y2 = (y1 - pad_y) / scale, (y2 - pad_y) / scale
        x0 = max(0, min(width, round(x1)))
        y0 = max(0, min(height, round(y1)))
        x_end = max(x0, min(width, round(x2)))
        y_end = max(y0, min(height, round(y2)))
        if x_end <= x0 or y_end <= y0:
            continue
        sku = detector.labels[class_id]
        products.append({"id": len(products) + 1, "sku": sku, "confidence": score,
                         "status": "ACCEPTED", "proposal_method": "keras_hub_retinanet",
                         "region_confidence": score, "x": x0, "y": y0,
                         "width": x_end - x0, "height": y_end - y0,
                         "image_width": width, "image_height": height, "image": image_id})
        cv2.rectangle(annotated, (x0, y0), (x_end, y_end), (50, 190, 70), max(2, width // 600))
        cv2.putText(annotated, f"{sku} {score:.0%}", (x0, max(18, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1600), (50, 190, 70), 2, cv2.LINE_AA)
    return {"annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "products": products,
            "processing": {"proposal_method": "keras_hub_retinanet", "candidate_regions": len(products),
                           "detector_loaded": True}}
