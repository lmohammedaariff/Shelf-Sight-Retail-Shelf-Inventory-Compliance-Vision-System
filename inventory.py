"""Visible facing summaries and evaluation metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def summarize_inventory(products: Sequence[Mapping]) -> dict:
    """Summarize detected visible facings, keeping unknowns explicit."""
    counts = Counter(str(item.get("sku", "Unknown product")) for item in products)
    unclassified = {"Unknown product", "Unclassified"}
    review_count = sum(
        1 for item in products
        if str(item.get("sku", "Unknown product")) in unclassified
        or str(item.get("status", "ACCEPTED")) in {"MODEL_REQUIRED", "REVIEW_REQUIRED"}
    )
    return {
        "visible_facings": len(products),
        "classified_facings": len(products) - review_count,
        "unknown_facings": review_count,
        "by_sku": [{"sku": sku, "facings": count} for sku, count in sorted(counts.items())],
    }


def inventory_metrics(ground_truth: Mapping[str, int], predicted: Mapping[str, int], tolerance: int = 0) -> dict:
    """Compute count MAE and within-tolerance accuracy over the union of SKUs."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    labels = sorted(set(ground_truth) | set(predicted))
    if any(int(value) < 0 for value in [*ground_truth.values(), *predicted.values()]):
        raise ValueError("Counts cannot be negative")
    if not labels:
        return {"sku_count": 0, "mae": 0.0, "within_tolerance_accuracy": 0.0, "per_sku": []}
    errors = [abs(int(predicted.get(sku, 0)) - int(ground_truth.get(sku, 0))) for sku in labels]
    return {
        "sku_count": len(labels),
        "mae": sum(errors) / len(errors),
        "within_tolerance_accuracy": sum(error <= tolerance for error in errors) / len(errors),
        "per_sku": [{"sku": sku, "ground_truth": int(ground_truth.get(sku, 0)),
                     "predicted": int(predicted.get(sku, 0)), "absolute_error": error}
                    for sku, error in zip(labels, errors)],
    }
