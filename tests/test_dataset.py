import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image

from pl_hot.dataset import prepare_seg_dataset_from_geojson, spatial_split
from pl_hot.geo_to_mask import rasterize_labels_for_chip
from pl_hot.params import PreprocessParams, SplitParams
from pl_hot.train import _SegDataset


def _first_split_with_chips(out_dir: Path) -> tuple[Path, Path]:
    for split in ("train", "val"):
        images = out_dir / split / "images"
        if list(images.glob("*.tif")):
            return images, out_dir / split / "masks"
    raise AssertionError("prepared dataset has no chips")


def test_spatial_split_keeps_blocks_together() -> None:
    names = [
        "OAM-0-0-18.tif",
        "OAM-1-0-18.tif",
        "OAM-8-0-18.tif",
        "OAM-9-0-18.tif",
    ]
    train, val = spatial_split(names, val_ratio=0.5, seed=0, block_size=4)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(names)
    for group in (["OAM-0-0-18.tif", "OAM-1-0-18.tif"], ["OAM-8-0-18.tif", "OAM-9-0-18.tif"]):
        in_train = all(n in train for n in group)
        in_val = all(n in val for n in group)
        assert in_train or in_val


def test_spatial_split_rejects_non_oam_names() -> None:
    names = ["chip_a.tif", "OAM-0-0-18.tif"]
    with pytest.raises(ValueError, match="chip_a.tif"):
        spatial_split(names, val_ratio=0.5, seed=42, block_size=4)


def test_prepare_seg_dataset_from_geojson(chips_dir, labels_dir, tmp_path) -> None:
    out_dir = tmp_path / "prepared"
    info = prepare_seg_dataset_from_geojson(
        chips_dir=chips_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        split_cfg=SplitParams(val_ratio=0.33, split_seed=7),
    )
    assert info["strategy"] == "spatial"
    assert info["train_count"] + info["val_count"] == 3
    assert info["train_count"] >= 1
    assert info["val_count"] >= 1
    assert info["labels_crs"] == "EPSG:4326"
    assert list((out_dir / "train" / "images").glob("*.tif"))
    assert list((out_dir / "val" / "images").glob("*.tif"))


def test_prepared_masks_match_native_chip_size(chips_dir, labels_dir, tmp_path) -> None:
    out_dir = tmp_path / "prepared"
    prepare_seg_dataset_from_geojson(
        chips_dir=chips_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        split_cfg=SplitParams(val_ratio=0.33, split_seed=7),
    )
    images_dir, masks_dir = _first_split_with_chips(out_dir)
    chip = next(images_dir.glob("*.tif"))
    with rasterio.open(chip) as src:
        height, width = src.height, src.width
    mask = Image.open(masks_dir / f"{chip.stem}.png").convert("L")
    assert mask.size == (width, height)
    arr = np.asarray(mask)
    assert arr.max() == 255
    assert arr.min() == 0
    # Parking square in the fixture is roughly pixels [8:20, 8:20] on a 32×32 chip.
    assert int((arr[8:20, 8:20] > 0).sum()) > 0


def test_prepare_rejects_single_spatial_block(chips_dir, labels_dir, tmp_path) -> None:
    same_block = tmp_path / "same_block"
    same_block.mkdir()
    for i, src in enumerate(sorted(chips_dir.glob("*.tif"))[:2]):
        shutil.copy2(src, same_block / f"OAM-{i:04d}-0000-18.tif")
    with pytest.raises(ValueError, match="empty"):
        prepare_seg_dataset_from_geojson(
            chips_dir=same_block,
            labels_dir=labels_dir,
            out_dir=tmp_path / "prepared",
            split_cfg=SplitParams(val_ratio=0.33, split_seed=7),
        )


def test_rasterize_reads_crs_from_geojson(chips_dir, tmp_path) -> None:
    chip = next(chips_dir.glob("*.tif"))
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [500008.0, 992.0],
                            [500008.0, 980.0],
                            [500020.0, 980.0],
                            [500020.0, 992.0],
                            [500008.0, 992.0],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }
    labels = tmp_path / "labels_3857.geojson"
    labels.write_text(json.dumps(geojson), encoding="utf-8")
    mask = rasterize_labels_for_chip(labels, chip)
    assert mask.shape == (32, 32)
    assert int(mask[8:20, 8:20].sum()) > 0


def test_seg_dataset_image_and_mask_share_model_size(chips_dir, labels_dir, tmp_path) -> None:
    out_dir = tmp_path / "prepared"
    prepare_seg_dataset_from_geojson(
        chips_dir=chips_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        split_cfg=SplitParams(val_ratio=0.33, split_seed=7),
    )
    images_dir, masks_dir = _first_split_with_chips(out_dir)
    ds = _SegDataset(
        images_dir,
        masks_dir,
        PreprocessParams(model_input_size=64, normalize_01=True, imagenet_norm=True),
    )
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (64, 64)
    assert y.dtype == np.float32
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert float(y.max()) == 1.0
    assert float(y.min()) == 0.0
    # ImageNet RGB can be negative; labels must stay 0/1 (not ImageNet-normalized).
    assert float(x.min()) < 0.0
