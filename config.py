"""Validated configuration types and safe defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageLimits:
    """Limits applied before decoding uploaded images into working arrays."""

    max_bytes: int = 20 * 1024 * 1024
    max_width: int = 8000
    max_height: int = 8000
    max_pixels: int = 30_000_000


@dataclass(frozen=True)
class VisionConfig:
    """Runtime settings for OpenCV proposals and SKU recognition."""

    resize_max_side: int = 1600
    max_regions: int = 100
    confidence_threshold: float = 0.65
    nms_iou_threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.resize_max_side < 128:
            raise ValueError("resize_max_side must be at least 128")
        if not 1 <= self.max_regions <= 1000:
            raise ValueError("max_regions must be between 1 and 1000")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 < self.nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be in (0, 1]")
