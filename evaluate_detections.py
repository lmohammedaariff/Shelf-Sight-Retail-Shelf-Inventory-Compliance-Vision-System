"""Evaluate class-aware boxes against test-split annotation JSONL at one IoU threshold."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfsight.dataset import validate_manifest
from shelfsight.metrics import detection_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSONL with image, class, normalized xywh bbox, and optional score")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/detection_metrics.json"))
    args = parser.parse_args()
    validation = validate_manifest(args.manifest, args.root)
    if not validation["valid"]:
        raise SystemExit("Invalid dataset manifest; run validate_dataset.py and fix errors first.")
    ground_truth = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] != "test":
            continue
        for obj in row["objects"]:
            if obj["type"] == "product":
                ground_truth.append({"image": row["image"], "class": obj["sku"], "bbox": obj["bbox"]})
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in predictions:
        if not all(key in row for key in ("image", "class", "bbox")) or not isinstance(row["bbox"], list) or len(row["bbox"]) != 4:
            raise SystemExit("Each prediction needs image, class, and normalized xywh bbox fields.")
    report = detection_metrics(ground_truth, predictions, args.iou)
    report["prediction_source"] = str(args.predictions)
    report["test_images"] = validation["split_images"].get("test", 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
