"""Create a reproducible exact-duplicate-grouped split from SKU image folders."""

import argparse
import json
from pathlib import Path

from shelfsight.dataset import create_classification_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/products"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/instances.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_classification_manifest(args.data, args.output, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
