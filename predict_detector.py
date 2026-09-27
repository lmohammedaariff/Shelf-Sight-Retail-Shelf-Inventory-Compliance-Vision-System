"""Write held-out split detector predictions in the evaluate_detections JSONL schema."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import numpy as np

from shelfsight.dataset import validate_manifest
from shelfsight.keras_hub_detector import detect_shelf, load_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/shelf_detector.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/detector_labels.json"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("reports/test_detector_predictions.jsonl"))
    args = parser.parse_args()
    validation = validate_manifest(args.manifest, args.root)
    if not validation["valid"]:
        raise SystemExit("Dataset manifest must validate before prediction.")
    detector = load_detector(args.model, args.labels)
    if detector is None:
        raise SystemExit("Detector model files are missing; train a detector first.")
    output = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] != "test":
            continue
        path = (args.root / row["image"]).resolve()
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"))
        predicted = detect_shelf(rgb, detector, threshold=args.threshold, image_id=row["image"])
        height, width = rgb.shape[:2]
        for product in predicted["products"]:
            output.append({"image": row["image"], "class": product["sku"],
                           "bbox": [product["x"] / width, product["y"] / height,
                                    product["width"] / width, product["height"] / height],
                           "score": product["confidence"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(f"Saved {len(output)} detections from {validation['split_images'].get('test', 0)} test images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
