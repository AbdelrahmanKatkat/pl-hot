from pl_hot.dataset import prepare_seg_dataset_from_geojson
from pl_hot.params import SplitParams


def test_prepare_seg_dataset_from_geojson(chips_dir, labels_dir, tmp_path) -> None:
    out_dir = tmp_path / "prepared"
    info = prepare_seg_dataset_from_geojson(
        chips_dir=chips_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        split_cfg=SplitParams(val_ratio=0.33, split_seed=7),
    )
    assert info["train_count"] + info["val_count"] == 3
    assert (out_dir / "train" / "images").exists()
    assert (out_dir / "train" / "masks").exists()
    assert (out_dir / "val" / "images").exists()
    assert (out_dir / "val" / "masks").exists()
    assert list((out_dir / "train" / "masks").glob("*.png"))
