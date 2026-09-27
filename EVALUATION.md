# Evaluation methodology

## Ground truth and split policy

All results must use annotated ground truth and a frozen held-out test split. Choose validation thresholds, early-stopping epochs, and calibration parameters using training/validation only. Group related frames and exact/near duplicates before splitting. Report the dataset version, license/source, class counts, image counts, split procedure, and known label limitations with every result.

## Available metrics

- **Classification:** accuracy, macro precision/recall/F1, per-class support/precision/recall/F1, confusion matrix, sample count, and prediction time as measured on the test pipeline.
- **Detection:** class-aware greedy matching at one chosen IoU threshold; precision, recall, F1, TP, FP, and FN. `mean_average_precision` is explicitly null because the current tool does not compute an AP curve over score thresholds/classes.
- **Inventory:** per-SKU absolute count error, MAE over the union of ground-truth and predicted SKU labels (including zero counts), and fraction within a user-defined absolute tolerance. Metrics refer to visible facings only.
- **Compliance:** rule-level slot results, correct image-level status rate, false-compliance count, false-violation count, and review coverage against independent image-level labels.
- **Latency:** training output reports prediction wall time over the held-out pipeline; the Streamlit UI reports a single image pipeline wall time. These are environment-specific and not a real-time guarantee.

## Commands

```powershell
python validate_dataset.py --manifest data/annotations/instances.jsonl --root data/products
python train.py --data data/products --manifest data/annotations/instances.jsonl --output models
python evaluation/evaluate_detections.py --manifest data/annotations/instances.jsonl --root data/products --predictions reports/test_predictions.jsonl
python evaluation/evaluate_inventory.py --records reports/inventory_test.jsonl
python evaluation/evaluate_compliance.py --planogram configs/example_planogram.json --records reports/compliance_test.jsonl
```

The evaluation command inputs are labeled records that the user must prepare from their test set; this repository does not contain them. Do not infer metrics from the sample/example planogram or synthetic unit tests.

## Interpretation and limitations

OpenCV proposals lack learned objectness scores, so precision/recall describe the proposal/classification pipeline only when predictions are saved with boxes and labels. The included app does not currently save a separate scored detection file automatically. F1 at one IoU threshold is not mAP. Counting is per visible face and is not a back-stock estimate. Compliance is tied to a chosen explicit slot map and independent ground truth. Unknown/review rows count as review outcomes, not successes.

Classification softmax scores are uncalibrated. To study calibration, reserve validation examples, compare reliability diagrams, Brier score and a stated ECE estimator, and tune acceptance thresholds on validation data. Report selective risk versus coverage and threshold-specific false acceptance; never use test data to set a cutoff.

## Bundled demo result

The bundled MobileNetV2 demo classifier was trained on 28 generated crops and evaluated on 7 held-out crops (one per class) from a single generated contact sheet. It scored 7/7 accuracy and 7/7 macro-F1 on that tiny synthetic split. The report is stored in `models/classification_metrics.json` and is shown in the app's Model & evaluation tab.

This result is a pipeline demonstration only. The training and holdout examples share one generated source sheet and do not represent store photography, different cameras, lighting, brands, or package variants. Do not present the score as real-world accuracy. No retail shelf detector, inventory count benchmark, or planogram compliance ground-truth evaluation is bundled. Current unit tests exercise code contracts on synthetic examples only and are not benchmark results.
