"""Chip preprocessing for SegFormer ONNX inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from .params import PreprocessParams

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class PreprocessMeta:
    width: int
    height: int
    scale_x: float
    scale_y: float
    transform: Any
    crs: Any


def preprocess_chip_for_onnx(path: str | Path, cfg: PreprocessParams) -> tuple[np.ndarray, PreprocessMeta]:
    """Load an RGB GeoTIFF chip and return NCHW float32 tensor and georef metadata."""
    image_path = Path(path)
    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    size = int(cfg.model_input_size)
    resized = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)  # CHW

    if cfg.normalize_01:
        arr = arr / 255.0
    if cfg.imagenet_norm:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    batch = arr[np.newaxis, ...].astype(np.float32)
    meta = PreprocessMeta(
        width=width,
        height=height,
        scale_x=width / float(size),
        scale_y=height / float(size),
        transform=transform,
        crs=crs,
    )
    return batch, meta
