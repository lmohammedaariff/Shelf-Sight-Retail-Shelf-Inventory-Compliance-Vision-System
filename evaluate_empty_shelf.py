"""Evaluate configured possible-empty slot IDs against annotated empty-slot IDs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfsight.metrics import empty_slot_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True,
                        help="JSONL: image_id, ground_truth_empty slot IDs, predicted_empty slot IDs")
    parser.add_argument("--output", type=Path, default=Path("reports/empty_slot_metrics.json"))
    args = parser.parse_args()
    per_image = []
    actual_all, predicted_all = [], []
    for line_no, line in enumerate(args.records.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        actual, predicted = row.get("ground_truth_empty"), row.get("predicted_empty")
        if not isinstance(actual, list) or not isinstance(predicted, list):
            raise SystemExit(f"Line {line_no}: ground_truth_empty and predicted_empty lists are required.")
        report = empty_slot_metrics(actual, predicted)
        per_image.append({"image_id": row.get("image_id", line_no), **report})
        actual_all.extend(f"{line_no}:{slot}" for slot in actual)
        predicted_all.extend(f"{line_no}:{slot}" for slot in predicted)
    result = {"images": len(per_image), "overall": empty_slot_metrics(actual_all, predicted_all),
              "per_image": per_image,
              "note": "Empty-slot detections remain review candidates; annotated ground truth is required for these metrics."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
