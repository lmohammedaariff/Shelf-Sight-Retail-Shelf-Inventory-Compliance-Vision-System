"""Evaluate visible-facing counts from ground-truth and prediction JSONL records."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfsight.inventory import inventory_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True,
                        help="JSONL rows with image_id, ground_truth counts, and predicted counts")
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("reports/inventory_metrics.json"))
    args = parser.parse_args()
    per_image = []
    aggregate_truth = {}
    aggregate_predicted = {}
    for line_no, line in enumerate(args.records.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("ground_truth"), dict) or not isinstance(row.get("predicted"), dict):
            raise SystemExit(f"Line {line_no}: ground_truth and predicted count maps are required.")
        report = inventory_metrics(row["ground_truth"], row["predicted"], tolerance=args.tolerance)
        per_image.append({"image_id": row.get("image_id", line_no), **report})
        for sku in set(row["ground_truth"]) | set(row["predicted"]):
            aggregate_truth[sku] = aggregate_truth.get(sku, 0) + int(row["ground_truth"].get(sku, 0))
            aggregate_predicted[sku] = aggregate_predicted.get(sku, 0) + int(row["predicted"].get(sku, 0))
    report = {"images": len(per_image), "aggregate_counts": inventory_metrics(aggregate_truth, aggregate_predicted,
              tolerance=args.tolerance), "per_image": per_image,
              "note": "Counts refer to visible facings in images, not total stock behind the front row."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
