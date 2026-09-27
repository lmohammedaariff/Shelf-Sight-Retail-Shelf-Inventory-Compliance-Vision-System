"""TensorFlow input pipeline for product boxes in annotation JSONL."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_product_crops(manifest_path: Path, data_root: Path) -> tuple[dict[str, list], list[str]]:
    """Load per-split product crop references and deterministic sorted SKU labels."""
    records: dict[str, list] = {"train": [], "validation": [], "test": []}
    classes: set[str] = set()
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        image_path = (data_root / record["image"]).resolve()
        image_path.relative_to(data_root.resolve())
        split = record["split"]
        for obj in record["objects"]:
            if obj.get("type") != "product":
                continue
            sku = obj["sku"].strip()
            classes.add(sku)
            records[split].append({"path": str(image_path), "bbox": obj["bbox"], "sku": sku,
                                   "image": record["image"], "line": line_no})
    labels = sorted(classes)
    for split, values in records.items():
        for row in values:
            row["class_id"] = labels.index(row["sku"])
    if len(labels) < 2:
        raise ValueError("Training needs at least two product SKU classes.")
    if any(not values for values in records.values()):
        raise ValueError("Manifest must have product crops in train, validation, and test splits.")
    for split in ("train", "validation", "test"):
        counts = Counter(item["sku"] for item in records[split])
        missing = sorted(set(labels) - set(counts))
        if missing:
            raise ValueError(f"Split {split!r} is missing classes: {', '.join(missing)}")
    return records, labels


def make_dataset(rows: list[dict], image_size: int, *, training: bool, seed: int):
    """Build a deterministic TensorFlow dataset of annotated product crops."""
    import tensorflow as tf

    paths = np.asarray([row["path"] for row in rows])
    boxes = np.asarray([row["bbox"] for row in rows], dtype=np.float32)
    labels = np.asarray([row["class_id"] for row in rows], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, boxes, labels))

    def load_crop(path, box, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        x, y, width, height = tf.unstack(box)
        patch = tf.image.crop_and_resize(tf.expand_dims(tf.cast(image, tf.float32), 0),
                                         tf.stack([[y, x, y + height, x + width]]), [0],
                                         [image_size, image_size])[0]
        return patch, label

    dataset = dataset.map(load_crop, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.1, seed=seed + 2),
            tf.keras.layers.RandomContrast(0.15, seed=seed + 3),
        ])
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(lambda image, label: (augmentation(image, training=True), label),
                              num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)
