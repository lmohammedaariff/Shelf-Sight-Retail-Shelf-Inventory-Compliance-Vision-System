"""Classical OpenCV product-region proposals (a baseline, not an object detector)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def intersection_over_union(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute standard intersection-over-union for xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return intersection / union if union else 0.0


def non_max_suppression(boxes: Sequence[Box], iou_threshold: float = 0.65) -> list[Box]:
    """Keep higher-area proposals while suppressing boxes with high IoU."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if box[2] <= 0 or box[3] <= 0:
            continue
        if all(intersection_over_union(box, prior) <= iou_threshold for prior in kept):
            kept.append(tuple(int(v) for v in box))
    return kept


def _shelf_band_fallback(rgb: np.ndarray, sensitivity: int, max_regions: int) -> list[Box]:
    """Split visually active shelf rows at persistent vertical package edges.

    This is a low-confidence classical CV fallback for images where contour
    extraction returns almost nothing. It is useful for front-facing shelves,
    but may split wide products or merge neighboring facings; it is not a
    learned object detector.
    """
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    vertical_energy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    activity = saturation.mean(axis=1) + 0.25 * vertical_energy.mean(axis=1)
    activity = np.convolve(activity, np.ones(5, dtype=np.float32) / 5, mode="same")
    row_threshold = max(25.0, float(np.percentile(activity, 40)) * 1.05)
    active_rows = activity >= row_threshold

    # Bridge small inactive gaps inside one shelf row.
    gap_limit = max(3, round(height * 0.025))
    row_groups: list[list[int]] = []
    for y in np.flatnonzero(active_rows):
        if not row_groups or y - row_groups[-1][-1] > gap_limit:
            row_groups.append([int(y)])
        else:
            row_groups[-1].append(int(y))

    boxes: list[Box] = []
    minimum_row_height = max(12, round(height * 0.10))
    padding_y = max(2, round(height * 0.035))
    minimum_width = max(5, round(width * 0.025))
    maximum_width = max(minimum_width + 1, round(width * 0.18))
    peak_percentile = max(60, min(85, 82 - 2 * sensitivity))
    peak_distance = max(4, round(width * 0.025))

    for group in row_groups:
        y0, y1 = group[0], group[-1] + 1
        if y1 - y0 < minimum_row_height:
            continue
        inner_y0, inner_y1 = min(y0 + 2, y1 - 1), max(y0 + 3, y1 - 2)
        edge_profile = np.abs(cv2.Sobel(gray[inner_y0:inner_y1], cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        edge_profile = np.convolve(edge_profile, np.ones(5, dtype=np.float32) / 5, mode="same")
        peak_threshold = float(np.percentile(edge_profile, peak_percentile))
        local_peaks: list[tuple[float, int]] = []
        for x in range(2, width - 2):
            if (edge_profile[x] >= edge_profile[x - 1]
                    and edge_profile[x] >= edge_profile[x + 1]
                    and edge_profile[x] > peak_threshold):
                local_peaks.append((float(edge_profile[x]), x))
        selected: list[int] = []
        for _, x in sorted(local_peaks, reverse=True):
            if all(abs(x - prior) >= peak_distance for prior in selected):
                selected.append(x)

        cuts = [0, *sorted(selected), width]
        intervals: list[tuple[int, int]] = []
        for left, right in zip(cuts, cuts[1:]):
            if right - left < minimum_width:
                continue
            pieces = max(1, int(np.ceil((right - left) / maximum_width)))
            boundaries = np.linspace(left, right, pieces + 1, dtype=int)
            intervals.extend((int(a), int(b)) for a, b in zip(boundaries, boundaries[1:])
                             if b - a >= minimum_width)

        box_y = max(0, y0 - padding_y)
        box_bottom = min(height, y1 + padding_y)
        boxes.extend((left, box_y, right - left, box_bottom - box_y)
                     for left, right in intervals)
        if len(boxes) >= max_regions:
            break

    return boxes[:max_regions]


def propose_regions(
    rgb: np.ndarray,
    sensitivity: int = 5,
    *,
    resize_max_side: int = 1600,
    max_regions: int = 100,
    nms_iou_threshold: float = 0.65,
) -> list[Box]:
    """Propose elongated foreground regions using CLAHE, edges, and contours.

    This baseline is sensitive to shelf geometry and may merge touching products,
    split textured packages, or propose shelf edges as objects.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an H×W×3 uint8 image")
    h, w = rgb.shape[:2]
    if not h or not w:
        return []
    if not 1 <= sensitivity <= 10:
        raise ValueError("sensitivity must be between 1 and 10")
    if resize_max_side < 128 or max_regions < 1:
        raise ValueError("resize_max_side and max_regions must be positive")
    scale = min(1.0, resize_max_side / max(h, w))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    low = 30 + (10 - sensitivity) * 3
    high = max(low + 20, 100 + (10 - sensitivity) * 5)
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = sh * sw * (0.00065 + sensitivity * 0.00010)
    candidates: list[Box] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < min_area or bw < sw * 0.025 or bh < sh * 0.10:
            continue
        if bw > sw * 0.80 or bh > sh * 0.90:
            continue
        aspect = bh / max(bw, 1)
        if not 0.45 <= aspect <= 8.0:
            continue
        px, py = int(bw * 0.035), int(bh * 0.02)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(sw, x + bw + px), min(sh, y + bh + py)
        candidates.append((round(x0 / scale), round(y0 / scale), round((x1 - x0) / scale), round((y1 - y0) / scale)))
    selected = non_max_suppression(candidates, nms_iou_threshold)
    contour_quality_is_low = (len(selected) < 3 or
                              (selected and float(np.median([box[3] for box in selected])) > sh * 0.28))
    if contour_quality_is_low:
        fallback = _shelf_band_fallback(small, sensitivity, max_regions)
        if scale < 1:
            fallback = [(round(x / scale), round(y / scale), round(bw / scale), round(bh / scale))
                        for x, y, bw, bh in fallback]
        if len(fallback) >= 3:
            selected = non_max_suppression(fallback, min(0.90, max(nms_iou_threshold, 0.80)))
    selected.sort(key=lambda b: (b[1] + b[3] / 2, b[0]))
    # A very wide contour is usually a merged shelf section or several adjacent
    # products. Keep those out of the per-facing classifier as one crop.
    selected = [box for box in selected if box[2] <= round(w * 0.25)]
    return selected[:max_regions]
