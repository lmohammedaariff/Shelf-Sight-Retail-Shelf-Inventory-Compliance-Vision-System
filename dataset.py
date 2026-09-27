"""Annotation-manifest validation and reproducible grouped dataset splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

SPLITS = {"train", "validation", "test"}


def _phash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unreadable image")
    if float(image.std()) < 2.0:
        # Perceptual hashes collapse all nearly uniform colors to similar values;
        # exact byte duplicates are already grouped by SHA-256.
        return int(_sha256(path)[:16], 16)
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    median = float(np.median(dct[1:]))
    value = 0
    for coefficient in dct:
        value = (value << 1) | int(coefficient > median)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest_path: Path, data_root: Path, *, near_duplicate_distance: int = 4) -> dict:
    """Validate a JSONL detection manifest, image files, boxes, labels, and split leakage."""
    errors: list[dict] = []
    warnings: list[dict] = []
    classes: Counter = Counter()
    split_images: Counter = Counter()
    hashes: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    records: list[dict] = []
    if not manifest_path.is_file():
        return {"valid": False, "errors": [{"line": None, "message": f"Manifest not found: {manifest_path}"}],
                "warnings": [], "images": 0, "classes": {}, "split_images": {}}
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_no, "message": f"Invalid JSON: {exc.msg}"})
            continue
        if not isinstance(record, dict):
            errors.append({"line": line_no, "message": "Each JSONL row must be an object."})
            continue
        image_rel = record.get("image")
        if not isinstance(image_rel, str) or not image_rel.strip():
            errors.append({"line": line_no, "message": "image must be a non-empty relative path."})
            continue
        pure_path = PurePosixPath(image_rel.replace("\\", "/"))
        if pure_path.is_absolute() or ".." in pure_path.parts:
            errors.append({"line": line_no, "message": "image path must stay inside the data root."})
            continue
        path = (data_root / Path(*pure_path.parts)).resolve()
        try:
            path.relative_to(data_root.resolve())
        except ValueError:
            errors.append({"line": line_no, "message": "image path resolves outside the data root."})
            continue
        if not path.is_file():
            errors.append({"line": line_no, "message": f"Image file is missing: {image_rel}"})
            continue
        split = record.get("split")
        if split not in SPLITS:
            errors.append({"line": line_no, "message": "split must be train, validation, or test."})
            continue
        try:
            with Image.open(path) as image:
                image.load()
                image_w, image_h = image.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            errors.append({"line": line_no, "message": f"Corrupt or unsupported image: {exc}"})
            continue
        try:
            digest = _sha256(path)
            perceptual = _phash(path)
        except Exception as exc:
            errors.append({"line": line_no, "message": f"Could not inspect image: {exc}"})
            continue
        hashes[digest].append((split, image_rel, line_no))
        split_images[split] += 1
        objects = record.get("objects")
        if not isinstance(objects, list):
            errors.append({"line": line_no, "message": "objects must be a list."})
            continue
        for obj_no, obj in enumerate(objects, start=1):
            prefix = f"object {obj_no}"
            if not isinstance(obj, dict):
                errors.append({"line": line_no, "message": f"{prefix} must be an object."})
                continue
            kind = obj.get("type")
            if kind not in {"product", "empty_space"}:
                errors.append({"line": line_no, "message": f"{prefix}.type must be product or empty_space."})
                continue
            if kind == "product":
                sku = obj.get("sku")
                if not isinstance(sku, str) or not sku.strip():
                    errors.append({"line": line_no, "message": f"{prefix}.sku is required for products."})
                else:
                    classes[sku.strip()] += 1
            elif obj.get("sku") is not None:
                errors.append({"line": line_no, "message": f"{prefix}.sku must be omitted for empty_space."})
            row_id = obj.get("shelf_row_id")
            if not isinstance(row_id, str) or not row_id.strip():
                warnings.append({"line": line_no, "message": f"{prefix} has no shelf_row_id."})
            box = obj.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                errors.append({"line": line_no, "message": f"{prefix}.bbox must be [x, y, width, height] normalized to 0..1."})
                continue
            try:
                x, y, width, height = [float(value) for value in box]
            except (TypeError, ValueError):
                errors.append({"line": line_no, "message": f"{prefix}.bbox values must be numeric."})
                continue
            if not all(np.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
                errors.append({"line": line_no, "message": f"{prefix}.bbox must be a positive box fully inside normalized image bounds."})
        record_copy = dict(record)
        record_copy["_line"] = line_no
        record_copy["_hash"] = digest
        record_copy["_phash"] = perceptual
        record_copy["_width"] = image_w
        record_copy["_height"] = image_h
        records.append(record_copy)
    for occurrences in hashes.values():
        if len({item[0] for item in occurrences}) > 1:
            errors.append({"line": occurrences[0][2], "message": "Exact duplicate image content occurs across dataset splits: " + ", ".join(item[1] for item in occurrences)})
    for left, right in _near_duplicate_pairs(records, near_duplicate_distance):
        if left["split"] != right["split"] and left["_hash"] != right["_hash"]:
            warnings.append({"line": left["_line"], "message": f"Possible near-duplicate split leakage: {left['image']} ↔ {right['image']} (review manually)."})
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "images": sum(split_images.values()), "classes": dict(sorted(classes.items())),
            "split_images": dict(sorted(split_images.items())),
            "duplicate_content_groups": sum(len(items) > 1 for items in hashes.values())}


def _near_duplicate_pairs(records: list[dict], maximum_distance: int):
    """Use eight-band perceptual-hash indexing to produce likely close pairs."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        value = record["_phash"]
        candidates: set[int] = set()
        for band in range(8):
            key = (band, (value >> (band * 8)) & 0xFF)
            candidates.update(buckets[key])
        for prior in candidates:
            pair = (prior, index)
            if pair in seen:
                continue
            seen.add(pair)
            if (record["_phash"] ^ records[prior]["_phash"]).bit_count() <= maximum_distance:
                yield records[prior], record
        for band in range(8):
            buckets[(band, (value >> (band * 8)) & 0xFF)].append(index)


def create_classification_manifest(data_root: Path, output_path: Path, *, seed: int = 42,
                                   ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict:
    """Build reproducible stratified train/validation/test records from class folders.

    Exact duplicate images are grouped before assignment, preventing identical
    bytes from crossing splits. Files are referenced; no dataset files are moved.
    """
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three positive fractions summing to 1")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    rng = random.Random(seed)
    records: list[dict] = []
    classes = [folder for folder in sorted(data_root.iterdir()) if folder.is_dir()]
    if len(classes) < 2:
        raise ValueError("At least two SKU subfolders are required.")
    grouped_near_duplicates = 0
    class_by_digest: dict[str, str] = {}
    for folder in classes:
        by_hash: dict[str, Path] = {}
        invalid_files: list[str] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                digest = _sha256(path)
            except (UnidentifiedImageError, OSError, ValueError):
                invalid_files.append(str(path))
                continue
            previous_class = class_by_digest.get(digest)
            if previous_class is not None and previous_class != folder.name:
                raise ValueError(f"Exact same image content occurs in conflicting SKU folders {previous_class!r} and {folder.name!r}: {path}")
            class_by_digest[digest] = folder.name
            by_hash.setdefault(digest, path)
        if invalid_files:
            raise ValueError(f"SKU {folder.name!r} contains unreadable images: {', '.join(invalid_files[:5])}")
        hashed_rows = [{"_path": path, "_hash": digest, "_phash": _phash(path)}
                       for digest, path in by_hash.items()]
        path_to_index = {row["_path"]: index for index, row in enumerate(hashed_rows)}
        parent = list(range(len(hashed_rows)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left, right in _near_duplicate_pairs(hashed_rows, 4):
            union(path_to_index[left["_path"]], path_to_index[right["_path"]])
        groups: dict[int, list[Path]] = defaultdict(list)
        for index, row in enumerate(hashed_rows):
            groups[find(index)].append(row["_path"])
        image_groups = list(groups.values())
        grouped_near_duplicates += sum(max(0, len(group) - 1) for group in image_groups)
        if len(image_groups) < 3:
            raise ValueError(f"SKU {folder.name!r} needs at least three independent image groups for train/validation/test; found {len(image_groups)}.")
        rng.shuffle(image_groups)
        n = len(image_groups)
        n_test = max(1, round(n * ratios[2]))
        n_validation = max(1, round(n * ratios[1]))
        while n_test + n_validation >= n:
            if n_test > 1:
                n_test -= 1
            else:
                n_validation -= 1
        labels = ["test"] * n_test + ["validation"] * n_validation + ["train"] * (n - n_test - n_validation)
        for group, split in zip(image_groups, labels):
            for path in group:
                records.append({"image": path.relative_to(data_root).as_posix(), "split": split,
                                "objects": [{"type": "product", "sku": folder.name,
                                             "bbox": [0.0, 0.0, 1.0, 1.0]}]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return {"records": len(records), "classes": len(classes), "seed": seed,
            "near_duplicate_images_grouped": grouped_near_duplicates,
            "split_counts": dict(Counter(record["split"] for record in records)),
            "manifest": str(output_path)}
