"""Validate an annotated JSONL dataset and emit real dataset statistics."""

import argparse
import json
from pathlib import Path

from shelfsight.dataset import validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/annotations/instances.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("data/products"),
                        help="Base directory for image paths in the annotation manifest")
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_validation.json"))
    args = parser.parse_args()
    report = validate_manifest(args.manifest, args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
