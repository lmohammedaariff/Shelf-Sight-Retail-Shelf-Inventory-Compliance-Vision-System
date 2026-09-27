"""Image decoding and validation helpers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ImageLimits


class ImageValidationError(ValueError):
    """Raised when an uploaded image is unsupported or exceeds safe limits."""


def decode_rgb_image(data: bytes, limits: ImageLimits | None = None) -> np.ndarray:
    """Decode image bytes to an EXIF-corrected RGB uint8 array after validation."""
    limits = limits or ImageLimits()
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > limits.max_bytes:
        raise ImageValidationError(f"Image exceeds the {limits.max_bytes // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ImageValidationError("Image dimensions must be positive.")
            if width > limits.max_width or height > limits.max_height:
                raise ImageValidationError(f"Image dimensions exceed {limits.max_width} × {limits.max_height} pixels.")
            if width * height > limits.max_pixels:
                raise ImageValidationError(f"Image exceeds the {limits.max_pixels:,}-pixel processing limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            return np.asarray(image, dtype=np.uint8)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc
