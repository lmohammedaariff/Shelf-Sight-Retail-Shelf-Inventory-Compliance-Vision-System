"""Backward-compatible facade for the refactored ShelfSight package."""

from shelfsight.classification import load_classifier
from shelfsight.compliance import evaluate_compliance, sequence_planogram
from shelfsight.detection import analyze_shelf as _analyze_shelf
from shelfsight.region_proposals import non_max_suppression, propose_regions


def analyze_shelf(rgb, classifier=None, manual_boxes=None, sensitivity=5, **kwargs):
    """Compatibility wrapper preserving the original public call signature."""
    return _analyze_shelf(rgb, classifier, manual_boxes=manual_boxes, sensitivity=sensitivity, **kwargs)


def make_planogram_report(products: list[dict], expected: list[str]) -> dict:
    """Compare legacy left-to-right product observations with expected SKU order."""
    if not expected:
        return {"compliance_pct": 0.0, "positions": [], "missing": [], "unexpected": [], "uncertain": []}
    planogram = sequence_planogram(expected)
    normalized = []
    count = max(1, len(products))
    for index, row in enumerate(products):
        item = dict(row)
        item.setdefault("x", index)
        item.setdefault("y", 0)
        item.setdefault("width", 1)
        item.setdefault("height", 1)
        item.setdefault("image_width", count)
        item.setdefault("image_height", 1)
        item.setdefault("status", "ACCEPTED" if item.get("sku") != "Unknown product" else "UNKNOWN_PRODUCT")
        normalized.append(item)
    report = evaluate_compliance(normalized, planogram)
    report["missing_skus"] = [row["expected"] for row in report["missing"]]
    return report
