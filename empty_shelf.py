"""Conservative empty-position candidates derived only from configured planogram slots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .planogram import Planogram


def possible_empty_slots(products: Sequence[Mapping], planogram: Planogram) -> list[dict]:
    """Flag configured slots without a nearby detection as possible, never confirmed, gaps.

    Missed detections and occlusion are indistinguishable from true empty positions
    in a single image, so every result requires human review.
    """
    results: list[dict] = []
    for row in planogram.rows:
        candidates = []
        for product in products:
            if not all(k in product for k in ("x", "y", "width", "height", "image_width", "image_height")):
                continue
            image_h = float(product["image_height"])
            image_w = float(product["image_width"])
            if image_h <= 0 or image_w <= 0:
                continue
            y = (float(product["y"]) + float(product["height"]) / 2) / image_h
            if abs(y - row.y_center) <= row.y_tolerance:
                x = (float(product["x"]) + float(product["width"]) / 2) / image_w
                candidates.append((x, product))
        for slot in row.slots:
            if not any(abs(x - slot.x_center) <= slot.tolerance for x, _ in candidates):
                results.append({"row_id": row.row_id, "slot_id": slot.slot_id,
                                "expected_sku": slot.expected_sku,
                                "status": "POSSIBLE_EMPTY_OR_MISSED_DETECTION",
                                "manual_review_required": True})
    return results
