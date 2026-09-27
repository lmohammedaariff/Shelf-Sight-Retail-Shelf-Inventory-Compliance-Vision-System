"""Unit tests for the image, geometry, inventory, and compliance contracts."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from shelfsight.compliance import evaluate_compliance, sequence_planogram
from shelfsight.detection import analyze_shelf, validate_manual_boxes
from shelfsight.inventory import inventory_metrics, summarize_inventory
from shelfsight.planogram import parse_planogram
from shelfsight.preprocessing import ImageValidationError, decode_rgb_image
from shelfsight.region_proposals import _shelf_band_fallback, intersection_over_union, non_max_suppression


class ImageTests(unittest.TestCase):
    def test_decode_valid_png(self):
        buffer = io.BytesIO()
        Image.new("RGB", (12, 8), (10, 20, 30)).save(buffer, format="PNG")
        decoded = decode_rgb_image(buffer.getvalue())
        self.assertEqual(decoded.shape, (8, 12, 3))
        self.assertEqual(decoded.dtype, np.uint8)

    def test_invalid_image_is_a_clear_error(self):
        with self.assertRaises(ImageValidationError):
            decode_rgb_image(b"this is not an image")

    def test_upload_byte_limit(self):
        with self.assertRaises(ImageValidationError):
            decode_rgb_image(b"12345", limits=type("Limits", (), {"max_bytes": 4})())


class GeometryTests(unittest.TestCase):
    def test_iou_and_nms(self):
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 5, 10, 10)), 25 / 175)
        boxes = non_max_suppression([(0, 0, 20, 20), (1, 1, 19, 19), (40, 0, 10, 10)])
        self.assertEqual(len(boxes), 2)

    def test_shelf_band_fallback_proposes_front_facing_products(self):
        image = np.full((180, 300, 3), 35, dtype=np.uint8)
        colors = [(220, 35, 35), (35, 180, 55), (240, 190, 25), (40, 100, 220),
                  (220, 65, 160), (30, 190, 190)]
        for row, (top, bottom) in enumerate(((18, 54), (67, 103), (116, 152))):
            for column, left in enumerate((10, 57, 104, 151, 198, 245)):
                cv2.rectangle(image, (left, top), (left + 37, bottom), colors[(column + row) % len(colors)], -1)
        boxes = _shelf_band_fallback(image, sensitivity=5, max_regions=100)
        self.assertGreaterEqual(len(boxes), 12)
        self.assertTrue(all(0 <= x < image.shape[1] and 0 <= y < image.shape[0]
                            and x + width <= image.shape[1] and y + height <= image.shape[0]
                            for x, y, width, height in boxes))

    def test_manual_boxes_are_clipped_only_when_valid(self):
        boxes = validate_manual_boxes([[0.1, 0.2, 0.3, 0.4]], 100, 100)
        self.assertEqual(boxes, [(10, 20, 30, 40)])
        with self.assertRaises(ValueError):
            validate_manual_boxes([[0.9, 0.2, 0.3, 0.4]], 100, 100)
        with self.assertRaises(ValueError):
            validate_manual_boxes({"bad": "schema"}, 100, 100)
        with self.assertRaises(ValueError):
            validate_manual_boxes([[float("nan"), 0, 0.2, 0.2]], 100, 100)

    def test_manual_end_to_end_no_model_is_explicitly_unclassified(self):
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        result = analyze_shelf(image, manual_boxes=[[0.1, 0.1, 0.3, 0.7]])
        self.assertEqual(len(result["products"]), 1)
        self.assertEqual(result["products"][0]["sku"], "Unclassified")
        self.assertEqual(result["products"][0]["status"], "MODEL_REQUIRED")
        self.assertIsNone(result["products"][0]["confidence"])
        self.assertIsNone(result["products"][0]["region_confidence"])


class InventoryTests(unittest.TestCase):
    def test_visible_and_unknown_counts(self):
        result = summarize_inventory([{"sku": "Cola"}, {"sku": "Unknown product"}, {"sku": "Cola"}])
        self.assertEqual(result["visible_facings"], 3)
        self.assertEqual(result["classified_facings"], 2)
        self.assertEqual(result["unknown_facings"], 1)

    def test_count_metrics_include_absent_sku(self):
        result = inventory_metrics({"A": 2, "B": 0}, {"A": 1, "C": 1}, tolerance=1)
        self.assertEqual(result["sku_count"], 3)
        self.assertAlmostEqual(result["mae"], 2 / 3)


class ComplianceTests(unittest.TestCase):
    def test_missing_duplicate_sku_quantity_is_reported(self):
        plan = sequence_planogram(["Cola", "Cola"])
        product = {"sku": "Cola", "status": "ACCEPTED", "x": 25, "y": 40,
                   "width": 20, "height": 20, "image_width": 100, "image_height": 100}
        report = evaluate_compliance([product], plan)
        self.assertEqual(len(report["missing"]), 1)
        self.assertEqual(report["compliance_pct"], 50.0)

    def test_unknown_prediction_requires_review(self):
        plan = sequence_planogram(["Cola"])
        unknown = {"sku": "Unknown product", "status": "REVIEW_REQUIRED", "x": 40, "y": 40,
                   "width": 20, "height": 20, "image_width": 100, "image_height": 100}
        report = evaluate_compliance([unknown], plan)
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["manual_review_recommended"], True)

    def test_json_planogram_schema_validation(self):
        data = {"shelf_id": "A", "rows": [{"row_id": "top", "y_center": .5,
                "slots": [{"slot_id": "one", "expected_sku": "Cola", "x_center": .5}]}]}
        self.assertEqual(parse_planogram(data).shelf_id, "A")
        data["rows"][0]["slots"][0]["x_center"] = 2
        with self.assertRaises(ValueError):
            parse_planogram(data)


if __name__ == "__main__":
    unittest.main()
