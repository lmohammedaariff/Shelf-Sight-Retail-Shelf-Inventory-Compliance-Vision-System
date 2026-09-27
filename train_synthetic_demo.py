"""Build and train the clearly marked synthetic seven-category demo SKU model.

This creates no real-world accuracy claim. The generated contact sheet provides
four synthetic examples per category; this is a demonstration/bootstrap model,
not a substitute for store-specific labeled product photos.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image
from shelfsight.metrics import classification_metrics

SHEET = ROOT / "data" / "synthetic_demo_contact_sheet.png"
SHELF = ROOT / "sample_shelf_upload.png"
DATA_ROOT = ROOT / "data" / "synthetic_demo_products"
MODEL_ROOT = ROOT / "models"
CLASSES = [
    "Coconut milk carton", "Tomato", "Carrot", "Green bell pepper",
    "Cucumber", "Yellow onion", "Cabbage",
]
IMAGE_SIZE = 224


def _save_sheet_crops() -> int:
    """Crop the exact 7x4 generated sheet, excluding its white gutters."""
    with Image.open(SHEET) as sheet:
        image = sheet.convert("RGB")
        width, height = image.size
        saved = 0
        for row in range(4):
            y0, y1 = round(row * height / 4), round((row + 1) * height / 4)
            for col, label in enumerate(CLASSES):
                x0, x1 = round(col * width / 7), round((col + 1) * width / 7)
                margin_x = max(4, round((x1 - x0) * 0.035))
                margin_y = max(4, round((y1 - y0) * 0.035))
                crop = image.crop((x0 + margin_x, y0 + margin_y,
                                   x1 - margin_x, y1 - margin_y))
                target = DATA_ROOT / label / f"synthetic-row-{row + 1}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target)
                saved += 1
    return saved


def _save_shelf_context_crops() -> int:
    """Add one shelf-context reference crop per product category."""
    boxes = {
        "Coconut milk carton": (0, 65, 505, 418),
        "Tomato": (500, 82, 1068, 416),
        "Carrot": (1060, 75, 1536, 416),
        "Green bell pepper": (0, 465, 365, 920),
        "Cucumber": (360, 465, 755, 920),
        "Yellow onion": (748, 465, 1156, 920),
        "Cabbage": (1148, 462, 1536, 920),
    }
    with Image.open(SHELF) as shelf:
        image = shelf.convert("RGB")
        saved = 0
        for label, box in boxes.items():
            target = DATA_ROOT / label / "synthetic-shelf-context.png"
            image.crop(box).save(target)
            saved += 1
    return saved


def _load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for class_id, label in enumerate(CLASSES):
        for path in sorted((DATA_ROOT / label).glob("*.png")):
            with Image.open(path) as source:
                crop = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                # Hold out the fourth contact-sheet row for an honest, small
                # synthetic-only check; never train on the examples scored here.
                if path.name == "synthetic-row-4.png":
                    test_images.append(np.asarray(crop, dtype=np.float32))
                    test_labels.append(class_id)
                else:
                    train_images.append(np.asarray(crop, dtype=np.float32))
                    train_labels.append(class_id)
    if not train_images or not test_images:
        raise RuntimeError("No generated demo crops were created.")
    return (np.stack(train_images), np.asarray(train_labels, dtype=np.int32),
            np.stack(test_images), np.asarray(test_labels, dtype=np.int32))


def main() -> int:
    if not SHEET.is_file() or not SHELF.is_file():
        raise FileNotFoundError("Expected generated demo assets are missing from the project.")
    count = _save_sheet_crops() + _save_shelf_context_crops()
    train_images, train_labels, test_images, test_labels = _load_dataset()
    tf.keras.utils.set_random_seed(42)

    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="rgb_crop_0_255")
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.12),
    ], name="light_augmentation")(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_scaling")(x)
    # The locally cached ImageNet MobileNetV2 artifact includes its original
    # 1000-class head. Reuse its pooled feature layer without any network fetch.
    imagenet_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=True, weights="imagenet")
    backbone = tf.keras.Model(imagenet_model.input, imagenet_model.layers[-2].output,
                              name="mobilenetv2_imagenet_features")
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="sku_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
        len(train_labels), seed=42, reshuffle_each_iteration=True).batch(8).prefetch(tf.data.AUTOTUNE)
    model.fit(train, epochs=45, verbose=2)
    test_probabilities = model.predict(test_images, verbose=0)
    metrics = classification_metrics(test_labels, test_probabilities.argmax(axis=1), CLASSES,
                                     probabilities=test_probabilities)
    metrics.update({
        "dataset_type": "synthetic_demo",
        "split": "contact-sheet rows 1-3 plus shelf-context crops for training; row 4 held out",
        "training_samples": int(len(train_labels)),
        "test_samples": int(len(test_labels)),
        "real_shelf_accuracy_claim": None,
        "limitations": "Tiny synthetic split from one generated contact sheet; not independent real-world validation.",
    })

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_ROOT / "sku_classifier.keras")
    (MODEL_ROOT / "labels.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    (MODEL_ROOT / "classification_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    info = {
        "architecture": "MobileNetV2 transfer learning",
        "synthetic_demo": True,
        "dataset": "AI-generated product contact sheet plus AI-generated shelf context crops",
        "dataset_image_count": len(train_labels),
        "dataset_total_image_count": len(train_labels) + len(test_labels),
        "synthetic_crop_count_written": count,
        "classes": CLASSES,
        "independent_held_out_evaluation": False,
        "synthetic_held_out_evaluation": True,
        "accuracy_claim": None,
        "limitations": [
            "Synthetic-only training data; not validated on real store photos.",
            "Scores are uncalibrated and predictions require human review.",
            "Add varied real product crops and retrain for useful store-specific results.",
        ],
    }
    (MODEL_ROOT / "model_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": str(MODEL_ROOT / "sku_classifier.keras"),
                      "labels": CLASSES, "training_images": len(train_labels),
                      "synthetic_test_images": len(test_labels),
                      "synthetic_crops_written": count,
                      "evaluation": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
