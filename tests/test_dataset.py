import pytest
from pl_hot.dataset import prepare_seg_dataset_from_geojson, spatial_split
from pl_hot.params import SplitParams


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
    assert (out_dir / "train" / "images").exists()
    assert (out_dir / "train" / "masks").exists()
    assert (out_dir / "val" / "images").exists()
    assert (out_dir / "val" / "masks").exists()
    assert list((out_dir / "train" / "masks").glob("*.png")) or list((out_dir / "val" / "masks").glob("*.png"))
