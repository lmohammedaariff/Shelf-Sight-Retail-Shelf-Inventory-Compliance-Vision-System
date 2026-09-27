"""Tests for dataset manifest validation and evaluation metrics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from shelfsight.dataset import validate_manifest
from shelfsight.metrics import classification_metrics, detection_metrics


class DatasetTests(unittest.TestCase):
    def test_validation_finds_duplicate_content_across_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (16, 16), (12, 25, 39)).save(root / "a.png")
            (root / "b.png").write_bytes((root / "a.png").read_bytes())
            rows = [
                {"image": "a.png", "split": "train", "objects": [{"type": "product", "sku": "A", "bbox": [0, 0, 1, 1]}]},
                {"image": "b.png", "split": "test", "objects": [{"type": "product", "sku": "A", "bbox": [0, 0, 1, 1]}]},
            ]
            manifest = root / "instances.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = validate_manifest(manifest, root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("duplicate image" in error["message"].lower() for error in result["errors"]))

    def test_rejects_bbox_outside_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (16, 16)).save(root / "a.png")
            row = {"image": "a.png", "split": "train", "objects": [{"type": "product", "sku": "A", "bbox": [.8, .8, .5, .5]}]}
            manifest = root / "instances.jsonl"
            manifest.write_text(json.dumps(row), encoding="utf-8")
            self.assertFalse(validate_manifest(manifest, root)["valid"])


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self):
        report = classification_metrics([0, 0, 1], [0, 1, 1], ["A", "B"])
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["confusion_matrix"], [[1, 1], [0, 1]])

    def test_detection_metrics_at_iou_threshold(self):
        truth = [{"image": "a", "class": "A", "bbox": [0, 0, 10, 10]}]
        preds = [{"image": "a", "class": "A", "bbox": [0, 0, 10, 10], "score": .9},
                 {"image": "a", "class": "B", "bbox": [0, 0, 10, 10], "score": .8}]
        report = detection_metrics(truth, preds)
        self.assertEqual(report["true_positive"], 1)
        self.assertEqual(report["false_positive"], 1)
        self.assertEqual(report["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
