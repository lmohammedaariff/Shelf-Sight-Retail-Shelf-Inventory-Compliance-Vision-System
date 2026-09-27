# ShelfSight — Retail Shelf & Inventory Compliance Vision System

ShelfSight is a local Streamlit research prototype for reviewing retail shelf photos. It combines classical OpenCV region proposals, optional TensorFlow SKU classification, visible-facing summaries, planogram slot checks, and explicit human-review states.

**Current project status:** the app includes a synthetic demo SKU classifier for seven broad product categories. It has a small 7-image synthetic holdout report, but no independent real-shelf evaluation. No real labeled shelf dataset or trained shelf detector is included. OpenCV proposals remain a classical baseline, not a learned object detector. See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) and the [evaluation notes](docs/EVALUATION.md).

## Features

- Validate image bytes, format, dimensions, and pixel limits before decoding.
- Propose candidate packages with OpenCV CLAHE, edge/contour filtering, and IoU non-maximum suppression; or supply manual rectangles.
- Load a locally trained MobileNetV2 SKU classifier. Low-score and unavailable-model cases remain `REVIEW_REQUIRED` / `UNKNOWN_PRODUCT` rather than being silently treated as a known SKU.
- Count visible candidate facings, retaining unknown counts. One image does not reveal behind-the-front stock.
- Compare one-row sequences or multi-row planogram JSON slots, including missing, unexpected, positional, unknown, and unassigned detections.
- Flag only configured slots without nearby detections as possible empty or missed detections; all require human review.
- Export slot-level CSV results when a planogram produces slot comparisons.
- Validate JSONL annotation datasets, detect exact cross-split duplicates, flag likely near-duplicate leakage, and report class/split counts.
- Train with a deterministic grouped split, validation/test metrics, checkpoints, class weights, training history, and confusion matrix.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit. The current project environment was exercised with Python 3.12, Streamlit 1.64, OpenCV 5.0, TensorFlow 2.21, NumPy 2.5, and pandas 3.0. Python 3.10–3.12 is targeted. If using a TensorFlow wheel unavailable for your operating system, use an officially supported TensorFlow environment and install the same project requirements there.

Images are processed in memory. Defaults limit uploads to 20 MB, 8,000 × 8,000 pixels, and 30 megapixels.

## Dataset workflow

### Option A: SKU crop folders for classification

Use one product crop per image with one class directory per SKU. Collect varied, licensed examples and avoid near-identical bursts across splits.

```text
data/products/
  Cola/cola_001.jpg
  Cola/cola_002.jpg
  Cola/cola_003.jpg
  Water/water_001.jpg
  Water/water_002.jpg
  Water/water_003.jpg
```

Make a reproducible train/validation/test manifest (exact and likely near-duplicate images are grouped before splitting), then validate it:

```powershell
python create_dataset_split.py --data data/products --output data/annotations/instances.jsonl --seed 42
python validate_dataset.py --manifest data/annotations/instances.jsonl --root data/products
```

At least three independent image groups per class are needed for a split; more data is strongly recommended. `create_dataset_split.py` treats each input image as one product crop with a full-image box. It is not a shelf-object annotation tool.

### Option B: Whole shelf photos with bounding boxes

Annotate licensed shelf photos using the JSONL schema in [DATASET.md](docs/DATASET.md). Include product class, normalized box, shelf row ID, and `train`/`validation`/`test` split. Add explicit `empty_space` annotations only where the image supports them. Group photos from the same burst/store/camera session into one split. Validate with:

```powershell
python validate_dataset.py --manifest data/annotations/instances.jsonl --root data
```

Whole-shelf annotations can train/evaluate classifier crops. An optional experimental KerasHub RetinaNet pipeline is also included for SKU-labeled shelf boxes; it requires the extra dependencies and substantial annotated data. It has not been trained or validated on retail data in this repository.

## Train and evaluate SKU classification

To rebuild the included synthetic demonstration model, run `python -m training.train_synthetic_demo`. It writes 35 generated crop examples under `data/synthetic_demo_products/`, trains on 28, holds out 7 contact-sheet crops, and writes the model and synthetic evaluation artifacts under `models/`. This is for a software demonstration only; it does not establish accuracy on real retail images.

```powershell
python train.py --data data/products --manifest data/annotations/instances.jsonl --epochs 20 --seed 42 --output models
```

Training writes `sku_classifier.keras`, `labels.json`, `model_info.json`, `training_history.csv`, `classification_metrics.json`, and `confusion_matrix.csv` under `models/`. The model embeds its pixel scaling so training and inference share the same preprocessing. The default ImageNet weights may need to be downloaded on first training; use `--weights none` to initialize from scratch when network access is unavailable. Test metrics are reported only after running against the manifest's held-out `test` split.

For optional whole-shelf RetinaNet training, first install `pip install -r requirements-detector.txt`, then run `python training/train_detector.py --manifest data/annotations/shelf_instances.jsonl --root data --output models`. The detector workflow remains experimental and needs SKU-labeled product boxes in each split. No detector model or measured detector result is bundled.

## Planogram JSON

The JSON mode in the sidebar accepts a schema like `configs/example_planogram.json`:

```json
{
  "shelf_id": "aisle-1",
  "rows": [{
    "row_id": "top",
    "y_center": 0.5,
    "slots": [
      {"slot_id": "left", "expected_sku": "Cola", "x_center": 0.25, "tolerance": 0.08},
      {"slot_id": "right", "expected_sku": "Water", "x_center": 0.75, "tolerance": 0.08}
    ]
  }]
}
```

Coordinates and tolerances are normalized to `[0, 1]`. A predicted row is assigned to the configured row whose normalized vertical center is nearest; there is no learned shelf-row detector. Slot occupancy is a candidate-based heuristic and every possible empty/missed slot is a human-review item.

## Evaluation and tests

```powershell
python -m unittest discover -s tests -v
python validate_dataset.py --manifest data/annotations/instances.jsonl --root data/products
```

Metrics are defined in [EVALUATION.md](docs/EVALUATION.md). The bundled classifier report scores 7/7 on a tiny synthetic holdout (one example per class); this is not a real-shelf benchmark. Detection evaluation implements class-aware precision/recall/F1 at one IoU threshold; it does not claim mAP. Inventory metrics measure visible facing counts against labeled counts. Compliance metrics need independently labeled image-level outcomes.

## Architecture and documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Dataset and annotation format](docs/DATASET.md)
- [Evaluation methodology](docs/EVALUATION.md)
- [Research plan and initial literature](docs/RESEARCH_PLAN.md)
- [Audit and current limitations](PROJECT_AUDIT.md)

## Known limitations

- OpenCV contours are sensitive to lighting, texture, shelf edges, touching packages, and perspective. They have no learned objectness score.
- An optional experimental TensorFlow/KerasHub RetinaNet workflow is included, but no detector is trained or validated on retail data here.
- The included seven-class SKU demo uses synthetic images only. It is not a substitute for a real, store-specific dataset and its scores are not validated.
- SKU softmax scores are not calibrated and do not guarantee out-of-distribution detection. The threshold is a review policy setting, not a probability guarantee.
- Empty positions inferred from expected planogram slots may be occluded or missed detections; the app does not call them confirmed stockouts.
- Planogram row/slot geometry must be configured. Complex package widths, perspective, and multiple stores are not automatically learned.
- Inventory counts are visible facings in the image, not stock behind them or back-room stock.
- Dataset provenance, licensing, store-held-out evaluation, model latency across deployment hardware, and research contribution need to be established for a real study.

## Project tree

```text
app.py                       Streamlit dashboard
shelfsight/                  Vision, data, inventory, planogram, reporting modules
training/                    TensorFlow data loader and training workflow
evaluation/                  Dataset-based metric command-line tools
tests/                       Unit tests for code contracts
configs/                     Example runtime and planogram configuration
data/                         User-provided data (ignored by Git except README)
models/                       Generated model artifacts (ignored by Git except README)
reports/                      Generated dataset/evaluation reports
```
