"""Backwards-compatible entry point for the reproducible classifier workflow."""

from training.train_classifier import main


if __name__ == "__main__":
    raise SystemExit(main())
