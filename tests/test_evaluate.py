from pathlib import Path

import numpy as np
from PIL import Image

from pl_hot.evaluate import evaluate_binary_masks


def _write_mask(path: Path, arr: np.ndarray) -> None:
    Image.fromarray((arr > 0).astype(np.uint8) * 255, mode="L").save(path)


def test_evaluate_binary_masks_reports_pw_and_miou(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()

    gt_arr = np.zeros((4, 4), dtype=np.uint8)
    gt_arr[1:3, 1:3] = 1
    pr_arr = gt_arr.copy()
    pr_arr[0, 0] = 1  # one FP pixel

    _write_mask(gt / "chip.png", gt_arr)
    _write_mask(pred / "chip.png", pr_arr)

    metrics = evaluate_binary_masks(pred, gt)
    assert metrics["accuracy"] == 15 / 16
    assert metrics["iou_parking_lot"] == 4 / 5
    assert metrics["iou_background"] == 11 / 12
    assert metrics["mean_iou"] == ((11 / 12) + (4 / 5)) / 2


def test_evaluate_does_not_inflate_empty_chips(tmp_path: Path) -> None:
    """Per-image mIoU would score an empty chip as 1.0 and pull the mean up."""
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()

    gt_arr = np.zeros((4, 4), dtype=np.uint8)
    gt_arr[1:3, 1:3] = 1
    pr_arr = gt_arr.copy()
    pr_arr[0, 0] = 1
    empty = np.zeros((4, 4), dtype=np.uint8)

    _write_mask(gt / "a.png", gt_arr)
    _write_mask(pred / "a.png", pr_arr)
    _write_mask(gt / "b.png", empty)
    _write_mask(pred / "b.png", empty)

    metrics = evaluate_binary_masks(pred, gt)
    assert metrics["accuracy"] == 31 / 32
    assert metrics["iou_parking_lot"] == 4 / 5
    assert metrics["iou_background"] == 27 / 28
    assert metrics["mean_iou"] < 0.9
