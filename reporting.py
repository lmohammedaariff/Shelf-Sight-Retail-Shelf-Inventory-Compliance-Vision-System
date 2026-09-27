"""JSON/CSV serialization for analysis and evaluation outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from io import StringIO


def to_json(data: Mapping) -> str:
    """Serialize a report to readable JSON while rejecting NaN/Infinity."""
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


def rows_to_csv(rows: Sequence[Mapping]) -> str:
    """Serialize homogeneous report rows to CSV."""
    if not rows:
        return ""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def detection_prediction_rows(products: Sequence[Mapping]) -> list[dict]:
    """Convert pixel-coordinate product records to normalized box metric rows."""
    results = []
    for row in products:
        image_w, image_h = float(row["image_width"]), float(row["image_height"])
        if image_w <= 0 or image_h <= 0:
            raise ValueError("Image dimensions must be positive for detection export")
        prediction = {"image": row.get("image", "uploaded-image"),
                      "class": row.get("sku", "Unclassified"),
                      "bbox": [float(row["x"]) / image_w, float(row["y"]) / image_h,
                               float(row["width"]) / image_w, float(row["height"]) / image_h],
                      "status": row.get("status", "REVIEW_REQUIRED")}
        if row.get("confidence", 1.0) is not None:
            prediction["score"] = float(row.get("confidence", 1.0))
        results.append(prediction)
    return results
