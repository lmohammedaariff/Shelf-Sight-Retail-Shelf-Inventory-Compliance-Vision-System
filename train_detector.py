"""Experimental KerasHub RetinaNet fine-tuning from shelf box annotations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Configure the backend and cache before TensorFlow/KerasHub are imported.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("KAGGLEHUB_CACHE", str(Path(__file__).resolve().parents[1] / ".cache" / "kagglehub"))

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfsight.dataset import validate_manifest


def read_detection_records(manifest: Path, root: Path) -> tuple[dict[str, list], list[str]]:
    """Group product boxes by source image; empty-space labels are excluded from SKU detection."""
    grouped: dict[str, dict] = {}
    labels: set[str] = set()
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        products = [obj for obj in record["objects"] if obj["type"] == "product"]
        if not products:
            continue
        image_path = (root / record["image"]).resolve()
        image_path.relative_to(root.resolve())
        item = grouped.setdefault(record["image"], {"path": str(image_path), "split": record["split"],
                                                      "boxes": [], "labels": [], "image_id": record["image"]})
        for product in products:
            sku = product["sku"].strip()
            labels.add(sku)
            x, y, width, height = product["bbox"]
            item["boxes"].append([x, y, x + width, y + height])
            item["labels"].append(sku)
    classes = sorted(labels)
    if len(classes) < 2:
        raise ValueError("Detector training needs at least two annotated product SKUs.")
    by_split = {split: [] for split in ("train", "validation", "test")}
    for row in grouped.values():
        by_split[row["split"]].append(row)
    for split in by_split:
        if not by_split[split]:
            raise ValueError(f"The manifest has no product images in the {split} split.")
    return by_split, classes


def make_detector_dataset(rows: list[dict], classes: list[str], image_size: int, batch_size: int,
                          *, training: bool, seed: int):
    """Build padded batches of square letterboxed images with absolute xyxy boxes."""
    class_to_id = {name: index for index, name in enumerate(classes)}

    def generator():
        for row_index, row in enumerate(rows):
            image = tf.io.decode_image(tf.io.read_file(row["path"]), channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            height = int(tf.shape(image)[0])
            width = int(tf.shape(image)[1])
            scale = min(image_size / width, image_size / height)
            new_w, new_h = round(width * scale), round(height * scale)
            resized = tf.image.resize(image, (new_h, new_w))
            pad_x, pad_y = (image_size - new_w) // 2, (image_size - new_h) // 2
            image = tf.image.pad_to_bounding_box(resized, pad_y, pad_x, image_size, image_size)
            boxes = tf.convert_to_tensor(row["boxes"], dtype=tf.float32)
            x1, y1, x2, y2 = tf.unstack(boxes, axis=1)
            boxes = tf.stack([x1 * width * scale + pad_x, y1 * height * scale + pad_y,
                              x2 * width * scale + pad_x, y2 * height * scale + pad_y], axis=1)
            labels = tf.convert_to_tensor([class_to_id[label] for label in row["labels"]], dtype=tf.int32)
            if len(row["boxes"]) > 100:
                raise ValueError(f"Image {row['image_id']} has more than 100 product boxes; crop or annotate within the configured limit.")
            if training:
                image, boxes = _random_horizontal_flip(image, boxes, image_size, seed + row_index)
                image = tf.image.random_brightness(image, max_delta=16.0, seed=seed + row_index)
                image = tf.clip_by_value(image, 0, 255)
            yield image / 255.0, boxes, labels

    signature = (tf.TensorSpec((image_size, image_size, 3), tf.float32),
                 tf.TensorSpec((None, 4), tf.float32), tf.TensorSpec((None,), tf.int32))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(
        [image_size, image_size, 3], [100, 4], [100]),
        padding_values=(0.0, -1.0, -1), drop_remainder=training)
    dataset = dataset.map(lambda image, boxes, labels: (image, {"boxes": boxes, "labels": labels}),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(max(8, len(rows)), seed=seed, reshuffle_each_iteration=True)
    return dataset.prefetch(tf.data.AUTOTUNE)


def _random_horizontal_flip(image, boxes, image_size: int, seed: int):
    """Flip both an image and its absolute xyxy boxes together."""
    should_flip = tf.random.stateless_uniform((), seed=[seed, 17]) < 0.5
    def flip():
        x1, y1, x2, y2 = tf.unstack(boxes, axis=1)
        return tf.image.flip_left_right(image), tf.stack([image_size - x2, y1, image_size - x1, y2], axis=1)
    return tf.cond(should_flip, flip, lambda: (image, boxes))


def main(argv: list[str] | None = None) -> int:
    try:
        import keras_hub
    except ImportError as exc:
        raise SystemExit("Optional detector dependencies are missing. Install requirements-detector.txt.") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/annotations/shelf_instances.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--pretrained-backbone", choices=["imagenet", "none"], default="imagenet")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 256:
        parser.error("epochs and batch-size must be positive; image-size must be at least 256")
    validation = validate_manifest(args.manifest, args.root)
    if not validation["valid"]:
        raise SystemExit("Dataset validation failed: " + "; ".join(row["message"] for row in validation["errors"][:10]))
    splits, labels = read_detection_records(args.manifest, args.root.resolve())
    tf.keras.utils.set_random_seed(args.seed)
    train_ds = make_detector_dataset(splits["train"], labels, args.image_size, args.batch_size,
                                     training=True, seed=args.seed)
    val_ds = make_detector_dataset(splits["validation"], labels, args.image_size, args.batch_size,
                                   training=False, seed=args.seed)
    if args.pretrained_backbone == "imagenet":
        image_encoder = keras_hub.models.Backbone.from_preset("resnet_50_imagenet")
    else:
        image_encoder = keras_hub.models.Backbone.from_preset("resnet_50_imagenet", load_weights=False)
    backbone = keras_hub.models.RetinaNetBackbone(image_encoder=image_encoder, min_level=3, max_level=5, use_p5=True)
    model = keras_hub.models.RetinaNetObjectDetector(backbone=backbone, num_classes=len(labels),
                bounding_box_format="xyxy", use_prediction_head_norm=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=10.0),
                  box_loss=tf.keras.losses.MeanAbsoluteError(reduction="sum"))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best_shelf_detector.weights.h5"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(args.output / "detector_training_history.csv")]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    if checkpoint.is_file():
        model.load_weights(checkpoint)
    model.save(args.output / "shelf_detector.keras")
    (args.output / "detector_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    info = {"architecture": "KerasHub RetinaNet", "backbone": "ResNet50", "class_count": len(labels),
            "labels": labels, "image_size": args.image_size, "seed": args.seed,
            "pretrained_backbone": args.pretrained_backbone,
            "dataset_manifest": str(args.manifest), "dataset_validation": validation,
            "evaluation_status": "Detector training completed; run evaluation/evaluate_detections.py with test predictions."}
    (args.output / "detector_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
