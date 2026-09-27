"""Train and evaluate a TensorFlow SKU classifier from annotated shelf crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfsight.dataset import create_classification_manifest, validate_manifest
from shelfsight.metrics import classification_metrics
from training.data_loader import load_product_crops, make_dataset


def build_model(num_classes: int, image_size: int, weights: str):
    """Build MobileNetV2 transfer learning with preprocessing embedded in the model."""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False,
        weights=None if weights == "none" else "imagenet")
    backbone.trainable = False
    inputs = tf.keras.Input((image_size, image_size, 3), name="rgb_crop_0_255")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_scaling")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="sku_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _evaluate(model, dataset, rows: list[dict], labels: list[str], output: Path) -> dict:
    start = time.perf_counter()
    probabilities = model.predict(dataset, verbose=0)
    elapsed = time.perf_counter() - start
    y_true = [row["class_id"] for row in rows]
    y_pred = np.argmax(probabilities, axis=1)
    metrics = classification_metrics(y_true, y_pred, labels, probabilities=probabilities)
    metrics["mean_batch_inference_ms"] = elapsed * 1000 / max(1, len(rows))
    metrics["confidence_note"] = "Softmax values are uncalibrated scores; do not interpret as calibrated probabilities."
    (output / "classification_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, metrics["confusion_matrix"]):
            writer.writerow([label, *row])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/products"),
                        help="Image root; by default one SKU subfolder per product")
    parser.add_argument("--manifest", type=Path, default=Path("data/annotations/instances.jsonl"))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    if args.image_size < 96 or args.image_size > 512 or args.epochs < 1:
        parser.error("image-size must be 96..512 and epochs must be positive")
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    if not args.manifest.is_file():
        try:
            result = create_classification_manifest(args.data, args.manifest, seed=args.seed)
            print(f"Created split manifest: {result}")
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"Could not create a reproducible data split: {exc}") from exc
    root = args.data.resolve()
    validation = validate_manifest(args.manifest, root)
    if not validation["valid"]:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(row["message"] for row in validation["errors"][:20]))
    print("Dataset summary:", json.dumps(validation, indent=2))
    records, labels = load_product_crops(args.manifest, root)
    counts = Counter(row["sku"] for row in records["train"])
    n = len(records["train"])
    class_weight = {index: n / (len(labels) * counts[label]) for index, label in enumerate(labels)}
    train_ds = make_dataset(records["train"], args.image_size, training=True, seed=args.seed)
    val_ds = make_dataset(records["validation"], args.image_size, training=False, seed=args.seed)
    test_ds = make_dataset(records["test"], args.image_size, training=False, seed=args.seed)
    model = build_model(len(labels), args.image_size, args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best_classifier.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(args.output / "training_history.csv"),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks,
              class_weight=class_weight, verbose=2)
    if checkpoint.exists():
        model = tf.keras.models.load_model(checkpoint)
    model.save(args.output / "sku_classifier.keras")
    (args.output / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    model_info = {"architecture": "MobileNetV2", "image_size": args.image_size,
                  "class_count": len(labels), "labels": labels, "seed": args.seed,
                  "weights": args.weights, "manifest": str(args.manifest), "dataset_validation": validation}
    (args.output / "model_info.json").write_text(json.dumps(model_info, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = _evaluate(model, test_ds, records["test"], labels, args.output)
    print("Held-out test metrics:", json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
