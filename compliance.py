"""Position-aware compliance checks and uncertainty status assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .planogram import Planogram


def evaluate_compliance(products: Sequence[Mapping], planogram: Planogram) -> dict:
    """Assign observations to the closest planogram row and compare row slots.

    Each observation uses pixel `x`, `y`, `width`, `height`, image_width and
    image_height. Missing detection and uncertain/unknown detections remain
    explicit; compliance is the fraction of expected slots with accepted SKU
    matches, not a calibrated probability or field-validated KPI.
    """
    row_observations: dict[str, list[Mapping]] = {row.row_id: [] for row in planogram.rows}
    unassigned: list[Mapping] = []
    for item in products:
        if not all(k in item for k in ("x", "y", "width", "height", "image_width", "image_height")):
            unassigned.append(item)
            continue
        image_h = float(item["image_height"])
        if image_h <= 0:
            unassigned.append(item)
            continue
        y = (float(item["y"]) + float(item["height"]) / 2) / image_h
        row = min(planogram.rows, key=lambda candidate: abs(candidate.y_center - y))
        if abs(row.y_center - y) <= row.y_tolerance:
            row_observations[row.row_id].append(item)
        else:
            unassigned.append(item)
    slot_results: list[dict] = []
    unexpected: list[dict] = []
    uncertain: list[dict] = []
    matched = 0
    expected_total = sum(len(row.slots) for row in planogram.rows)
    for row in planogram.rows:
        observations = sorted(row_observations[row.row_id], key=lambda item: float(item.get("x", 0)))
        available = set(range(len(observations)))
        for slot in row.slots:
            if not available:
                slot_results.append({"shelf_id": planogram.shelf_id, "row_id": row.row_id,
                                     "slot_id": slot.slot_id, "expected": slot.expected_sku,
                                     "detected": "Missing detection", "status": "MISSING"})
                continue
            image_w = float(observations[0].get("image_width", 0))
            if image_w <= 0:
                candidates = list(available)
                selected = candidates[0]
                distance = None
            else:
                selected = min(available, key=lambda idx: abs(
                    (float(observations[idx].get("x", 0)) + float(observations[idx].get("width", 0)) / 2) / image_w - slot.x_center))
                distance = abs((float(observations[selected].get("x", 0)) +
                                float(observations[selected].get("width", 0)) / 2) / image_w - slot.x_center)
            item = observations[selected]
            available.remove(selected)
            sku = str(item.get("sku", "Unknown product"))
            status = str(item.get("status", "ACCEPTED" if sku != "Unknown product" else "UNKNOWN_PRODUCT"))
            if status != "ACCEPTED" or sku == "Unknown product":
                decision = "REVIEW_REQUIRED"
                uncertain.append({"row_id": row.row_id, "slot_id": slot.slot_id, "sku": sku, "status": status})
            elif sku != slot.expected_sku:
                decision = "MISMATCH"
            elif distance is not None and distance > slot.tolerance:
                decision = "POSITION_MISMATCH"
            else:
                decision = "MATCH"
                matched += 1
            slot_results.append({"shelf_id": planogram.shelf_id, "row_id": row.row_id,
                                 "slot_id": slot.slot_id, "expected": slot.expected_sku,
                                 "detected": sku, "position_error": distance, "status": decision})
        for idx in sorted(available):
            item = observations[idx]
            unexpected.append({"row_id": row.row_id, "sku": item.get("sku", "Unknown product"),
                               "x": item.get("x"), "y": item.get("y"), "status": "UNEXPECTED"})
    review_rows = [dict(item, status="UNASSIGNED_ROW") for item in unassigned]
    uncertain.extend(review_rows)
    denom = expected_total or 1
    return {
        "shelf_id": planogram.shelf_id,
        "status": "REVIEW_REQUIRED" if uncertain else ("COMPLIANT" if matched == expected_total and not unexpected else "NON_COMPLIANT"),
        "matched_slots": matched,
        "expected_slots": expected_total,
        "compliance_pct": 100.0 * matched / denom,
        "positions": slot_results,
        "missing": [row for row in slot_results if row["status"] == "MISSING"],
        "unexpected": unexpected,
        "uncertain": uncertain,
        "manual_review_recommended": bool(uncertain),
    }


def sequence_planogram(expected: Sequence[str], shelf_id: str = "shelf-1") -> Planogram:
    """Convert the legacy left-to-right SKU sequence into one normalized row."""
    from .planogram import PlanogramRow, PlanogramSlot

    items = [str(sku).strip() for sku in expected]
    if not items or any(not sku for sku in items):
        raise ValueError("Expected sequence must contain non-empty SKU names.")
    count = len(items)
    slots = tuple(PlanogramSlot(f"slot-{i + 1}", sku, (i + 0.5) / count, 0.5 / count)
                  for i, sku in enumerate(items))
    return Planogram(shelf_id=shelf_id, rows=(PlanogramRow("row-1", 0.5, slots, 0.5),))
