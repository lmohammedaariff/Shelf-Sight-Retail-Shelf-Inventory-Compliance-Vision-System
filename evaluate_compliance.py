"""Evaluate slot-based planogram decisions against labeled image-level outcomes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfsight.compliance import evaluate_compliance
from shelfsight.planogram import parse_planogram


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True,
                        help="JSONL: image_id, products, ground_truth_status")
    parser.add_argument("--output", type=Path, default=Path("reports/compliance_metrics.json"))
    args = parser.parse_args()
    planogram = parse_planogram(json.loads(args.planogram.read_text(encoding="utf-8")))
    records = []
    for line_no, line in enumerate(args.records.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("products"), list) or row.get("ground_truth_status") not in {"COMPLIANT", "NON_COMPLIANT", "REVIEW_REQUIRED"}:
            raise SystemExit(f"Line {line_no}: products and a valid ground_truth_status are required.")
        records.append({"image_id": row.get("image_id", line_no),
                        "ground_truth": row["ground_truth_status"],
                        "prediction": evaluate_compliance(row["products"], planogram)})
    false_compliance = sum(row["prediction"]["status"] == "COMPLIANT" and row["ground_truth"] != "COMPLIANT" for row in records)
    false_violation = sum(row["prediction"]["status"] == "NON_COMPLIANT" and row["ground_truth"] == "COMPLIANT" for row in records)
    correct = sum(row["prediction"]["status"] == row["ground_truth"] for row in records)
    report = {"images": len(records), "rule_correctness": correct / len(records) if records else 0.0,
              "false_compliance_count": false_compliance, "false_violation_count": false_violation,
              "manual_review_count": sum(row["prediction"]["status"] == "REVIEW_REQUIRED" for row in records),
              "records": records,
              "note": "Metrics are meaningful only when ground_truth_status is independently annotated on a held-out set."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
