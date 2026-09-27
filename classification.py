"""Keras model loading and confidence-aware SKU classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SKUClassifier:
    """Validated Keras SKU classifier and its class-name mapping."""

    model: object
    labels: list[str]
    model_path: Path


@dataclass(frozen=True)
class SKUResult:
    """Top class, raw score, and decision status for a crop."""

    sku: str
    confidence: float | None
    status: str


def load_classifier(model_path: Path, labels_path: Path) -> SKUClassifier | None:
    """Load and validate a saved Keras classifier; log and raise invalid artifacts."""
    if not model_path.exists() and not labels_path.exists():
        return None
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Both classifier model and labels.json must exist.")
    import tensorflow as tf

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) < 2 or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("Classifier labels must be a list of at least two non-empty strings.")
    model = tf.keras.models.load_model(model_path)
    if len(model.output_shape) != 2 or model.output_shape[-1] != len(labels):
        raise ValueError("Model output count does not match labels.json.")
    if len(model.input_shape) != 4 or model.input_shape[-1] != 3:
        raise ValueError("Classifier must accept a batch of three-channel images.")
    return SKUClassifier(model=model, labels=labels, model_path=model_path)


def classify_crop(crop_rgb: np.ndarray, classifier: SKUClassifier | None, threshold: float = 0.65) -> SKUResult:
    """Classify a crop using the same MobileNetV2 scaling used by the trainer.

    The score is a raw softmax output, not a calibrated probability. Crops below
    the configured acceptance threshold retain their top-class suggestion but
    stay in review; only unusable scores are returned as unknown.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3 or crop_rgb.size == 0:
        raise ValueError("crop_rgb must be a non-empty H×W×3 image")
    if classifier is None:
        return SKUResult("Unclassified", None, "MODEL_REQUIRED")
    shape = classifier.model.input_shape
    height, width = shape[1], shape[2]
    height = int(height or 160)
    width = int(width or 160)
    resized = cv2.resize(crop_rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    # The trainer embeds MobileNetV2 scaling in the saved model graph. Passing
    # raw RGB [0,255] here applies the same transform used during training.
    batch = np.expand_dims(resized, axis=0)
    probabilities = classifier.model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    score = float(probabilities[index])
    if not np.isfinite(score):
        return SKUResult("Unknown product", 0.0, "REVIEW_REQUIRED")
    if score < threshold:
        return SKUResult(classifier.labels[index], score, "REVIEW_REQUIRED")
    return SKUResult(classifier.labels[index], score, "ACCEPTED")
