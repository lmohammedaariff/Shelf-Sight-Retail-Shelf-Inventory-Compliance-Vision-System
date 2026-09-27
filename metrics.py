"""Evaluation metrics that do not require synthetic or ungrounded results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str],
                           probabilities: Sequence[Sequence[float]] | None = None,
                           ece_bins: int = 15) -> dict:
    """Compute classification scores and, when supplied, Brier/ECE/selective metrics."""
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if true.shape != predicted.shape or true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal length")
    if np.any(true < 0) or np.any(predicted < 0) or np.any(true >= len(labels)) or np.any(predicted >= len(labels)):
        raise ValueError("Class index outside labels")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(true, predicted):
        matrix[actual, guess] += 1
    per_class = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "support": int(matrix[index, :].sum()),
                          "precision": precision, "recall": recall, "f1": f1})
    n = int(len(true))
    report = {"samples": n, "accuracy": float(np.mean(true == predicted)) if n else 0.0,
            "macro_precision": float(np.mean([row["precision"] for row in per_class])) if per_class else 0.0,
            "macro_recall": float(np.mean([row["recall"] for row in per_class])) if per_class else 0.0,
            "macro_f1": float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0,
            "labels": list(labels), "per_class": per_class, "confusion_matrix": matrix.tolist()}
    if probabilities is not None:
        scores = np.asarray(probabilities, dtype=np.float64)
        if scores.shape != (n, len(labels)) or not np.isfinite(scores).all():
            raise ValueError("probabilities must be finite with shape [samples, classes]")
        if not 2 <= ece_bins <= 100:
            raise ValueError("ece_bins must be between 2 and 100")
        confidence = scores.max(axis=1) if n else np.asarray([], dtype=float)
        guessed = scores.argmax(axis=1) if n else np.asarray([], dtype=int)
        correct = guessed == true
        ece = 0.0
        bins = []
        for index in range(ece_bins):
            lower, upper = index / ece_bins, (index + 1) / ece_bins
            selected = (confidence >= lower) & ((confidence < upper) if index < ece_bins - 1 else (confidence <= upper))
            count = int(selected.sum())
            if count:
                accuracy = float(correct[selected].mean())
                mean_confidence = float(confidence[selected].mean())
                ece += count / max(1, n) * abs(accuracy - mean_confidence)
                bins.append({"lower": lower, "upper": upper, "count": count,
                             "accuracy": accuracy, "mean_confidence": mean_confidence})
        one_hot = np.eye(len(labels), dtype=np.float64)[true] if n else np.zeros_like(scores)
        report["confidence_metrics"] = {
            "expected_calibration_error": ece,
            "multiclass_brier_score": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))) if n else 0.0,
            "ece_bins": ece_bins, "reliability_bins": bins,
            "note": "ECE depends on binning and estimates calibration on this evaluation split; it is not a per-example guarantee.",
        }
        report["selective_metrics"] = []
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            accepted = confidence >= threshold
            count = int(accepted.sum())
            report["selective_metrics"].append({"threshold": threshold, "accepted": count,
                "coverage": count / max(1, n),
                "accepted_accuracy": float(correct[accepted].mean()) if count else None})
    return report


def empty_slot_metrics(ground_truth_empty: Sequence[str], predicted_empty: Sequence[str]) -> dict:
    """Compute slot-level empty-position precision/recall against annotated ground truth."""
    actual, predicted = set(ground_truth_empty), set(predicted_empty)
    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "manual_review_required": True}


def detection_metrics(ground_truth: Sequence[dict], predictions: Sequence[dict], iou_threshold: float = 0.5) -> dict:
    """Compute single-threshold class-aware precision/recall using greedy IoU matches."""
    from .region_proposals import intersection_over_union

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    unmatched = set(range(len(ground_truth)))
    true_positive = 0
    for prediction in sorted(predictions, key=lambda item: float(item.get("score", 1.0)), reverse=True):
        candidates = []
        for index in unmatched:
            actual = ground_truth[index]
            if prediction.get("image") != actual.get("image") or prediction.get("class") != actual.get("class"):
                continue
            overlap = intersection_over_union(prediction["bbox"], actual["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, index))
        if candidates:
            _, selected = max(candidates)
            unmatched.remove(selected)
            true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {"iou_threshold": iou_threshold, "true_positive": true_positive,
            "false_positive": false_positive, "false_negative": false_negative,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_average_precision": None,
            "note": "Single IoU-threshold precision/recall; mAP requires scored detections across IoU thresholds/classes."}
