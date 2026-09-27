# Dataset and annotation protocol

## Bundled demonstration data

The repository contains a small AI-generated synthetic demo dataset for seven broad classes (coconut milk carton, tomato, carrot, green bell pepper, cucumber, yellow onion, and cabbage). It is only used by `training/train_synthetic_demo.py`; the script holds out one generated contact-sheet crop per class for a tiny synthetic-only evaluation. These are not real retail images, the holdout is not independent real-world validation, and it supports no real-world accuracy claim. The repository still contains no real product photographs or annotated shelf boxes. Collect store/product images with permission and confirm the source license allows the intended research, processing, storage, and redistribution before use.

Potential public research resources include the SKU-110K packed-retail dataset and the RPC checkout dataset. Their task domains, annotation schemas, access requirements, and licenses must be inspected at their original project pages before using or redistributing them. Checkout/individual-product images are not automatically equivalent to shelf-audit data.

## Recommended folder layout

```text
data/
  shelf_images/
    store_001/...
  annotations/
    instances.jsonl
  products/
    SKU_A/...
    SKU_B/...
```

The class-folder split command is for isolated product crops only. It treats each file as a single SKU crop and assigns its full extent as the box. For whole shelves, annotate boxes with a dedicated tool and provide the documented JSONL rows.

## JSONL annotation schema

Each line is one image. `image` is a path relative to the `--root` argument. `split` is one of `train`, `validation`, or `test`. `objects` may include `product` or explicitly annotated `empty_space` regions.

```json
{
  "image": "shelf_images/store_001/frame_0001.jpg",
  "split": "train",
  "store_id": "store_001",
  "capture_session": "visit_2026_01",
  "objects": [
    {"type": "product", "sku": "SKU_A", "shelf_row_id": "row_1", "bbox": [0.10, 0.20, 0.18, 0.62]},
    {"type": "empty_space", "shelf_row_id": "row_1", "bbox": [0.29, 0.25, 0.08, 0.50]}
  ]
}
```

`bbox` is `[x, y, width, height]` normalized against the original image, each coordinate from 0 to 1. Boxes must have positive extent and remain inside the image. Product boxes need a non-empty SKU. Empty-space annotations omit `sku`. `shelf_row_id` is recommended and its absence is a validation warning. Additional metadata is preserved but not interpreted by the training loader.

For unknown or unrecognized product packages, annotate the visible object as a product with a consistent reserved label such as `__unknown__`; do not silently omit it. Do not label an occluded gap as empty unless the annotation policy supports that conclusion. Document whether labels describe visible package facings, shelf slots, or stockout events.

## Splitting and leakage controls

`create_dataset_split.py` sorts folders/files, uses a fixed seed, groups exact duplicates and likely near duplicates within each SKU, then assigns groups to train/validation/test at approximately 80/10/10. At least three independent groups per class are required; a small dataset cannot support a reliable held-out estimate. When manually annotating shelf photos, assign all frames from the same burst, store visit, shelf, or highly related camera session to one split. Prefer a store-held-out test set for domain generalization.

The validator checks missing files, image decode, split values, object types, normalized boxes, product labels, exact duplicate bytes across splits, likely perceptual-hash near duplicates, image/split counts, and SKU annotation counts. Near-duplicate warnings require human adjudication. Class counts refer to product annotations, not necessarily independent image counts.

Run:

```powershell
python create_dataset_split.py --data data/products --output data/annotations/instances.jsonl --seed 42
python validate_dataset.py --manifest data/annotations/instances.jsonl --root data/products
```

For a whole-shelf annotation root, specify that root explicitly to `validate_dataset.py` (for example `--root data`). If duplicates leak across manually created splits, change the split assignments and re-run validation. Never move images into a split after inspecting test performance.

## Acquisition and annotation checklist

- Capture multiple stores, visits, lighting, angles, occlusions, crowding, empty positions, product variants, package updates, and unsupported products.
- Keep labels consistent at the intended SKU granularity; record ambiguous package variants for review.
- Use independent annotators for a subset and resolve label disagreements before reporting metrics.
- Preserve original images and version annotation revisions.
- Record source, license/permissions, store/session identifier, image dimensions, annotation tool/version, split rule, and label taxonomy.
- Avoid faces, payment screens, customer identifiers, or other private content; crop/redact when permitted and necessary.
- Dataset quality and model performance cannot be judged from the codebase alone.
