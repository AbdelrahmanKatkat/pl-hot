"""Prepare segmentation datasets from chips + one GeoJSON labels file."""

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geo_to_mask import rasterize_labels_for_chip
from .params import SplitParams

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def spatial_split(
    chip_names: list[str],
    val_ratio: float,
    seed: int,
    *,
    block_size: int = 4,
) -> tuple[list[str], list[str]]:
    """Block-spatial split on fAIr OAM tile coords; entire `(x//K, y//K)` blocks pick a side.

    fAIr chips are always `OAM-{x}-{y}-{z}.tif` (see fAIr-models sample layout).
    """
    blocks: dict[tuple[int, int], list[str]] = {}
    for name in chip_names:
        oam_match = OAM_TILE_RE.match(name)
        if oam_match is None:
            raise ValueError(f"Expected OAM-{{x}}-{{y}}-{{z}} chip name, got {name!r}")
        tile_x, tile_y = int(oam_match.group(1)), int(oam_match.group(2))
        blocks.setdefault((tile_x // block_size, tile_y // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    block_keys = sorted(blocks.keys())
    rng.shuffle(block_keys)

    n_total = len(chip_names)
    n_val_target = max(1, int(n_total * val_ratio)) if n_total else 0

    val: list[str] = []
    train: list[str] = []
    for key in block_keys:
        bucket = val if len(val) < n_val_target else train
        bucket.extend(blocks[key])

    return sorted(train), sorted(val)


def _find_labels_geojson(labels_dir: str | Path) -> Path:
    root = Path(labels_dir)
    matches = sorted(root.glob("*.geojson"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson in {root}, found {len(matches)}")
    return matches[0]


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

    train_names, val_names = spatial_split(
        [p.name for p in chip_paths],
        split_cfg.val_ratio,
        split_cfg.split_seed,
        block_size=split_cfg.block_size,
    )

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
        "strategy": "spatial",
        "val_ratio": split_cfg.val_ratio,
        "seed": split_cfg.split_seed,
        "block_size": split_cfg.block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
        "labels_geojson": str(labels_geojson),
    }
