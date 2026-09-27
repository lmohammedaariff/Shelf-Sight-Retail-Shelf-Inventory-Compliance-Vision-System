"""Streamlit ShelfSight interface for shelf analysis, inventory, and compliance."""

from __future__ import annotations

import json
import hashlib
import io
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from shelfsight.classification import load_classifier
from shelfsight.compliance import evaluate_compliance, sequence_planogram
from shelfsight.config import ImageLimits, VisionConfig
from shelfsight.detection import analyze_shelf
from shelfsight.empty_shelf import possible_empty_slots
from shelfsight.inventory import summarize_inventory
from shelfsight.keras_hub_detector import detect_shelf, load_detector
from shelfsight.planogram import parse_planogram
from shelfsight.preprocessing import ImageValidationError, decode_rgb_image
from shelfsight.reporting import rows_to_csv

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sku_classifier.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
MODEL_INFO_PATH = ROOT / "models" / "model_info.json"
REPORTS_PATH = ROOT / "reports"
logger = logging.getLogger("shelfsight.app")


@st.cache_resource(show_spinner="Loading SKU model…")
def get_classifier(model_mtime: float | None):
    """Cache the optional classifier by model modification time."""
    del model_mtime
    return load_classifier(MODEL_PATH, LABELS_PATH)


@st.cache_resource(show_spinner="Loading TensorFlow detector…")
def get_detector(model_mtime: float | None):
    """Cache the optional KerasHub detector by model modification time."""
    del model_mtime
    return load_detector(ROOT / "models" / "shelf_detector.keras",
                         ROOT / "models" / "detector_labels.json",
                         ROOT / "models" / "detector_info.json")


def save_verified_product_crops(image_rgb: np.ndarray, products: list[dict]) -> tuple[int, int, int]:
    """Persist human-reviewed product regions into the SKU training folders.

    Returns (saved, duplicates, skipped). Folder names become the model's SKU
    labels, so filesystem-reserved characters are replaced with underscores.
    """
    data_root = ROOT / "data" / "products"
    saved = duplicates = skipped = 0
    for product in products:
        label = str(product.get("sku", "")).strip()
        if product.get("label_source") != "human" or not label or label.casefold() in {
            "unclassified", "unknown product", "unknown", "n/a"
        }:
            skipped += 1
            continue
        class_name = re.sub(r"[<>:\"/\\|?*]+", "_", label).strip(" .")
        if not class_name or class_name in {".", ".."}:
            skipped += 1
            continue
        if class_name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            class_name += "_SKU"
        x, y = int(product["x"]), int(product["y"])
        width, height = int(product["width"]), int(product["height"])
        crop = image_rgb[y:y + height, x:x + width]
        if crop.size == 0:
            skipped += 1
            continue
        buffer = io.BytesIO()
        Image.fromarray(crop, mode="RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        output = class_dir / f"{digest}.png"
        if output.exists():
            duplicates += 1
        else:
            output.write_bytes(image_bytes)
            saved += 1
    return saved, duplicates, skipped


def product_training_coverage(data_root: Path) -> list[dict]:
    """Summarize crop counts per SKU for the in-app training guidance."""
    if not data_root.is_dir():
        return []
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [{"SKU": folder.name,
             "Labeled crops": sum(1 for path in folder.rglob("*")
                                  if path.is_file() and path.suffix.lower() in valid_suffixes)}
            for folder in sorted(data_root.iterdir()) if folder.is_dir()]


st.set_page_config(page_title="ShelfSight | Retail vision", page_icon="🛒", layout="wide")
st.title("🛒 ShelfSight")
st.caption("Retail shelf and inventory compliance · OpenCV region proposals + optional TensorFlow SKU model")

with st.sidebar:
    st.header("Shelf analysis")
    detector_model_path = ROOT / "models" / "shelf_detector.keras"
    detection_modes = ["Automatic (OpenCV)", "Manual rectangles"]
    if detector_model_path.is_file():
        detection_modes.insert(1, "TensorFlow RetinaNet")
    detection_mode = st.radio("Product regions", detection_modes,
                              help="OpenCV proposals are a classical baseline. RetinaNet appears here only after a labeled shelf-box detector has been trained.")
    if not detector_model_path.is_file():
        st.caption("RetinaNet detector is not trained; automatic mode uses OpenCV region proposals.")
    sensitivity = st.slider("Region proposal sensitivity", 1, 10, 5)
    confidence_threshold = st.slider("SKU acceptance threshold", 0.0, 1.0, 0.65, 0.01,
                                     help="Scores are uncalibrated softmax outputs; below-threshold crops are sent to review.")
    manual_box_text = st.text_area("Manual boxes (normalized [x, y, width, height])",
                                   value="[[0.05, 0.2, 0.16, 0.65], [0.23, 0.2, 0.16, 0.65]]",
                                   disabled=detection_mode != "Manual rectangles")
    st.divider()
    planogram_mode = st.radio("Planogram input", ["SKU sequence", "JSON configuration"], horizontal=True)
    if planogram_mode == "SKU sequence":
        sequence_text = st.text_area("Expected SKU order (left to right)", value="",
                                     placeholder="Cola\nWater\nJuice",
                                     help="One SKU per slot in one row. For multiple rows, choose JSON configuration.")
        planogram_text = ""
    else:
        sequence_text = ""
        planogram_text = st.text_area("Planogram JSON", value="", height=220,
            placeholder='{"shelf_id":"A","rows":[{"row_id":"top","y_center":0.5,"slots":[{"slot_id":"left","expected_sku":"Cola","x_center":0.25},{"slot_id":"right","expected_sku":"Water","x_center":0.75}]}]}')

uploaded = st.file_uploader("Upload a front-facing shelf photo", type=["jpg", "jpeg", "png", "webp"],
                           help="Maximum 20 MB, 8000 × 8000 pixels, 30 megapixels.")
image_review_id = "no-image"
if uploaded is not None:
    if uploaded.size > ImageLimits().max_bytes:
        st.error("This image exceeds the 20 MB limit.")
        image_rgb = None
    else:
        try:
            image_bytes = uploaded.getvalue()
            image_rgb = decode_rgb_image(image_bytes, ImageLimits())
            image_review_id = hashlib.sha256(image_bytes).hexdigest()[:16]
        except ImageValidationError as exc:
            st.error(str(exc))
            image_rgb = None
else:
    image_rgb = None

planogram = None
plan_error = None
if planogram_mode == "SKU sequence" and sequence_text.strip():
    expected_sequence = [line.strip() for line in sequence_text.splitlines() if line.strip()]
    try:
        planogram = sequence_planogram(expected_sequence)
    except ValueError as exc:
        plan_error = str(exc)
elif planogram_mode == "JSON configuration" and planogram_text.strip():
    try:
        planogram = parse_planogram(json.loads(planogram_text))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        plan_error = f"Invalid planogram: {exc}"
if plan_error:
    st.error(plan_error)

classifier = None
detector = None
model_error = None
demo_model = False
if image_rgb is not None:
    try:
        if detection_mode == "TensorFlow RetinaNet":
            detector_path = ROOT / "models" / "shelf_detector.keras"
            detector = get_detector(detector_path.stat().st_mtime if detector_path.is_file() else None)
            if detector is None:
                st.error("TensorFlow RetinaNet model is not installed. Train it with `python training/train_detector.py` after adding annotated shelf images.")
        else:
            model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.is_file() else None
            classifier = get_classifier(model_mtime)
            if classifier and MODEL_INFO_PATH.is_file():
                try:
                    demo_model = bool(json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8")).get("synthetic_demo"))
                except (OSError, json.JSONDecodeError, AttributeError):
                    demo_model = False
    except Exception as exc:
        logger.exception("SKU model could not be loaded")
        model_error = str(exc)
    if model_error:
        st.error(f"SKU model error: {model_error}. Analysis can continue, but SKU labels will be unavailable until the model files are fixed.")

analysis = None
compliance = None
empty_candidates = []
inventory = None
processing_ms = None
if image_rgb is not None:
    manual_boxes = None
    if detection_mode == "Manual rectangles":
        try:
            manual_boxes = json.loads(manual_box_text)
        except json.JSONDecodeError as exc:
            st.error(f"Manual regions must be valid JSON: {exc.msg}")
    if (detection_mode != "Manual rectangles" or manual_boxes is not None) and not (detection_mode == "TensorFlow RetinaNet" and detector is None):
        started = time.perf_counter()
        try:
            if detection_mode == "TensorFlow RetinaNet":
                analysis = detect_shelf(image_rgb, detector, threshold=confidence_threshold, image_id=uploaded.name)
            else:
                analysis = analyze_shelf(image_rgb, classifier, manual_boxes=manual_boxes,
                    sensitivity=sensitivity, threshold=confidence_threshold,
                    config=VisionConfig(confidence_threshold=confidence_threshold), image_id=uploaded.name)
            processing_ms = (time.perf_counter() - started) * 1000
            inventory = summarize_inventory(analysis["products"])
            if planogram is not None:
                compliance = evaluate_compliance(analysis["products"], planogram)
                empty_candidates = possible_empty_slots(analysis["products"], planogram)
        except (ValueError, TypeError, OverflowError) as exc:
            st.error(f"Analysis input is invalid: {exc}")
        except Exception as exc:
            logger.exception("Shelf analysis failed")
            st.error(f"Shelf analysis failed: {exc}")

analysis_tab, inventory_tab, compliance_tab, evaluation_tab, settings_tab = st.tabs(
    ["Shelf analysis", "Inventory", "Planogram compliance", "Model & evaluation", "Dataset & settings"])

with analysis_tab:
    if analysis is None:
        st.caption("The annotated image and per-product decisions appear here after successful analysis.")
    else:
        if analysis["products"]:
            if classifier is None:
                st.info("No trained SKU model is loaded. These regions were not classified. Train with labeled product photos for automatic names, or enter a verified name manually.")
            review_rows = [{"id": row["id"], "sku": row["sku"],
                            "model_score": (f'{row["confidence"]:.3f}' if row["confidence"] is not None else "N/A"),
                            "status": row["status"]}
                           for row in analysis["products"]]
            reviewed = st.data_editor(
                pd.DataFrame(review_rows),
                key=f"product-name-review-{image_review_id}",
                hide_index=True,
                use_container_width=True,
                disabled=["id", "model_score", "status"],
                column_config={
                    "id": st.column_config.NumberColumn("Region", format="%d"),
                    "sku": st.column_config.TextColumn(
                        "SKU / product name", required=True,
                        help="Type a name such as Coconut milk, Tomato, or Carrot."
                    ),
                    "model_score": st.column_config.TextColumn("Model score"),
                    "status": st.column_config.TextColumn("Decision"),
                },
            )
            for product, edited in zip(analysis["products"], reviewed.to_dict(orient="records")):
                edited_name = str(edited.get("sku", "")).strip()
                if edited_name and edited_name != product["sku"]:
                    product["sku"] = edited_name
                    product["status"] = "ACCEPTED"
                    product["label_source"] = "human"
            human_labeled = [row for row in analysis["products"] if row.get("label_source") == "human"]
            if human_labeled:
                st.caption("Only save a region after checking that its box contains one product and the name is correct. Each class needs several distinct product photos before training.")
                if st.button("Save verified product crops for training", key=f"save-labels-{image_review_id}"):
                    try:
                        saved, duplicates, skipped = save_verified_product_crops(image_rgb, analysis["products"])
                        st.success(f"Saved {saved} new crop(s); {duplicates} duplicate(s) already existed; skipped {skipped} unverified or invalid region(s).")
                        st.info("Collect varied photos for every SKU. The training split requires at least three independent images per SKU; more is strongly recommended. Then run the training steps shown in Dataset & settings.")
                    except OSError as exc:
                        st.error(f"Could not save training crops: {exc}")
            inventory = summarize_inventory(analysis["products"])
            if planogram is not None:
                compliance = evaluate_compliance(analysis["products"], planogram)
                empty_candidates = possible_empty_slots(analysis["products"], planogram)

        image_column, summary_column = st.columns([1.35, 1])
        with image_column:
            st.image(analysis["annotated"], caption="Candidate boxes; green means accepted classifier result, orange means review/unknown.", use_container_width=True)
        with summary_column:
            first, second, third = st.columns(3)
            first.metric("Visible candidate facings", inventory["visible_facings"])
            second.metric("Classified facings", inventory["classified_facings"])
            third.metric("Processing time", f"{processing_ms:.0f} ms")
            st.metric("Recognition model", "RetinaNet" if detection_mode == "TensorFlow RetinaNet" and detector else ("Synthetic demo" if demo_model else ("SKU classifier" if classifier else "Not trained")))
            if demo_model:
                st.caption("Synthetic demo classes: coconut milk carton, tomato, carrot, green bell pepper, cucumber, yellow onion, and cabbage. Scores are not calibrated; review names manually.")
            elif classifier:
                st.caption("OpenCV proposal boxes do not have a learned objectness score. The SKU model's softmax score is uncalibrated.")
            else:
                st.caption("OpenCV finds candidate regions only. Train a SKU classifier with labeled product photos to get product names and model scores.")
        if not analysis["products"]:
            st.warning("No candidate regions were proposed. Try a straight-on, well-lit image or enter manual rectangles.")

with inventory_tab:
    if inventory is None:
        st.info("Upload and analyze an image to see visible-facing counts.")
    else:
        a, b, c = st.columns(3)
        a.metric("Visible candidate facings", inventory["visible_facings"])
        b.metric("Classified facings", inventory["classified_facings"])
        c.metric("Unknown / review", inventory["unknown_facings"])
        if inventory["by_sku"]:
            st.dataframe(pd.DataFrame(inventory["by_sku"]), hide_index=True, use_container_width=True)
        st.caption("These are visible detections from one image, not total on-shelf or back-room inventory. Empty-position flags require a planogram and human review.")
        if empty_candidates:
            st.warning(f"Possible empty or missed detections: {len(empty_candidates)} configured slot(s) need visual review.")
            st.dataframe(pd.DataFrame(empty_candidates), hide_index=True, use_container_width=True)
        elif planogram is None:
            st.info("Add a planogram to identify expected slots that have no nearby detected facing.")

with compliance_tab:
    if planogram is None:
        st.info("Enter a SKU sequence or JSON planogram configuration in the sidebar.")
    elif compliance is None:
        st.info("Upload and analyze a shelf image to compare it with the planogram.")
    else:
        st.metric("Slot compliance", f"{compliance['compliance_pct']:.1f}%")
        if compliance["status"] == "COMPLIANT":
            st.success("All configured slots matched in this analysis.")
        elif compliance["status"] == "REVIEW_REQUIRED":
            st.warning("Some observations are uncertain or could not be assigned. Review before deciding compliance.")
        else:
            st.error("The observed arrangement does not match all configured slots.")
        if compliance["positions"]:
            st.dataframe(pd.DataFrame(compliance["positions"]), hide_index=True, use_container_width=True)
        if compliance["missing"]:
            st.subheader("Missing detections")
            st.dataframe(pd.DataFrame(compliance["missing"]), hide_index=True, use_container_width=True)
        if compliance["unexpected"]:
            st.subheader("Unexpected detections")
            st.dataframe(pd.DataFrame(compliance["unexpected"]), hide_index=True, use_container_width=True)
        if compliance["uncertain"]:
            st.subheader("Human review")
            st.dataframe(pd.DataFrame(compliance["uncertain"]), hide_index=True, use_container_width=True)
        st.caption("The compliance score compares assigned candidate boxes with the configured slot rules. It is not field-validated and depends on the proposal and SKU models.")

with evaluation_tab:
    st.subheader("Model evaluation")
    metrics_path = ROOT / "models" / "classification_metrics.json"
    model_info = {}
    if MODEL_INFO_PATH.is_file():
        try:
            model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            model_info = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("accuracy") is not None:
                left, middle, right = st.columns(3)
                left.metric("Held-out accuracy", f"{metrics['accuracy']:.1%}")
                middle.metric("Test images", metrics.get("samples", "—"))
                right.metric("Macro F1", f"{metrics['macro_f1']:.1%}" if metrics.get("macro_f1") is not None else "—")
                if model_info.get("synthetic_demo") or metrics.get("dataset_type") == "synthetic_demo":
                    st.warning("This report uses synthetic demo images only. It does not estimate accuracy on real shelf photos.")
                with st.expander("Full evaluation details"):
                    st.json(metrics)
            else:
                st.warning("An evaluation file exists, but it has no held-out accuracy metrics. Regenerate it from a labeled test set.")
                with st.expander("Evaluation file details"):
                    st.json(metrics)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the classification report: {exc}")
    else:
        is_demo = bool(model_info.get("synthetic_demo"))
        st.warning("Evaluation not run: no labeled held-out test set is available.")
        if is_demo:
            st.markdown(
                f"The loaded **{model_info.get('architecture', 'SKU classifier')}** is trained on "
                f"**{model_info.get('dataset_image_count', '—')} synthetic crops** across "
                f"**{len(model_info.get('classes', []))} categories**. A training score would only show "
                "how well it memorized those demo examples, so it is intentionally not presented as evaluation."
            )
            st.caption("Demo categories: " + ", ".join(model_info.get("classes", [])))
        st.markdown("**To create a useful evaluation report**")
        st.markdown(
            "1. In **Dataset & settings**, add several real product photos for each SKU, with varied views and lighting.  "
            "\n2. Keep some separately photographed examples out of training for the test set.  "
            "\n3. Train and evaluate using that split; the report will show held-out metrics here."
        )
    dataset_report = REPORTS_PATH / "dataset_validation.json"
    if dataset_report.is_file():
        try:
            st.subheader("Dataset validation report")
            st.json(json.loads(dataset_report.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the dataset report: {exc}")
    st.caption("Held-out metrics are only meaningful when test photos are labeled and were not used to train the model.")

with settings_tab:
    st.subheader("Runtime configuration")
    st.json({"region_method": "OpenCV contour baseline", "confidence_threshold": confidence_threshold,
             "image_limits": {"max_upload_mb": ImageLimits().max_bytes // (1024 * 1024),
                              "max_width": ImageLimits().max_width,
                              "max_height": ImageLimits().max_height,
                              "max_pixels": ImageLimits().max_pixels},
             "model_present": MODEL_PATH.is_file() and LABELS_PATH.is_file(),
             "detector_present": (ROOT / "models" / "shelf_detector.keras").is_file(),
             "model_version": "0.2.0"})
    if MODEL_INFO_PATH.is_file():
        try:
            current_model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
            if current_model_info.get("synthetic_demo"):
                st.success("SKU classifier trained and loaded: synthetic seven-category demo model.")
                st.warning("This model used generated sample crops. It has not been independently evaluated on real shelf photos, so review every prediction.")
        except (OSError, json.JSONDecodeError):
            pass
    if not (ROOT / "models" / "shelf_detector.keras").is_file():
        st.info("RetinaNet shelf detector: not trained. Automatic (OpenCV) mode uses classical region proposals; training RetinaNet requires several shelf images with product bounding-box labels.")
    st.subheader("Dataset status")
    demo_data_root = ROOT / "data" / "synthetic_demo_products"
    demo_coverage = product_training_coverage(demo_data_root)
    if demo_coverage:
        demo_count = sum(row["Labeled crops"] for row in demo_coverage)
        st.success(f"Synthetic demo dataset: {demo_count} labeled crops across {len(demo_coverage)} categories.")
        st.dataframe(pd.DataFrame(demo_coverage), hide_index=True, use_container_width=True)
    product_data_root = ROOT / "data" / "products"
    coverage = product_training_coverage(product_data_root)
    if coverage:
        st.dataframe(pd.DataFrame(coverage), hide_index=True, use_container_width=True)
        if len(coverage) >= 2 and all(row["Labeled crops"] >= 3 for row in coverage):
            st.code("python create_dataset_split.py --data data/products --output data/annotations/instances.jsonl --seed 42\npython validate_dataset.py --manifest data/annotations/instances.jsonl --root data/products\npython train.py --data data/products --manifest data/annotations/instances.jsonl --epochs 20 --seed 42 --output models", language="powershell")
        else:
            st.warning("Add at least 3 independent product photos for every SKU and at least 2 different SKUs before preparing a training split. Photos should vary in angle, lighting, and background.")
    else:
        st.info("No additional human-labeled store photos are saved yet. The loaded classifier is the synthetic demo model shown above.")
        st.caption("To replace the demo with a store-specific model, label verified regions and save varied real product crops, then use the training steps shown here.")

if analysis is not None:
    if compliance and compliance["positions"]:
        st.download_button("Download slot results CSV", rows_to_csv(compliance["positions"]),
                           file_name="planogram-slots.csv", mime="text/csv")
