"""Prepare segmentation datasets from chips + one GeoJSON labels file."""

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geo_to_mask import rasterize_labels_for_chip
from .params import SplitParams


def _find_labels_geojson(labels_dir: str | Path) -> Path:
    root = Path(labels_dir)
    matches = sorted(root.glob("*.geojson"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson in {root}, found {len(matches)}")
    return matches[0]


def _split_names(chip_names: list[str], cfg: SplitParams) -> tuple[list[str], list[str]]:
    rng = random.Random(cfg.split_seed)
    names = chip_names.copy()
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * cfg.val_ratio)))
    n_val = min(n_val, max(1, len(names) - 1)) if len(names) > 1 else 1
    val = sorted(names[:n_val])
    train = sorted(names[n_val:])
    return train, val


def prepare_seg_dataset_from_geojson(
    chips_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    split_cfg: SplitParams,
) -> dict[str, Any]:
    """Create image/mask train-val folders and return split metadata."""
    chips_root = Path(chips_dir)
    out_root = Path(out_dir)
    labels_geojson = _find_labels_geojson(labels_dir)

    chip_paths = sorted(list(chips_root.glob("*.tif")) + list(chips_root.glob("*.tiff")))
    if len(chip_paths) < 2:
        raise ValueError("Need at least 2 chips for train/val split")

    train_names, val_names = _split_names([p.name for p in chip_paths], split_cfg)

    for split_name, names in (("train", train_names), ("val", val_names)):
        img_dir = out_root / split_name / "images"
        mask_dir = out_root / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for name in names:
            src = chips_root / name
            dst = img_dir / name
            dst.write_bytes(src.read_bytes())
            mask = rasterize_labels_for_chip(labels_geojson, src)
            Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(mask_dir / f"{src.stem}.png")

    return {
        "strategy": "random",
        "val_ratio": split_cfg.val_ratio,
        "seed": split_cfg.split_seed,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
        "labels_geojson": str(labels_geojson),
    }
