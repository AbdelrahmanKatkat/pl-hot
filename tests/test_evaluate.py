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
    assert 0.0 <= metrics["fair:accuracy"] <= 1.0
    assert 0.0 <= metrics["fair:mean_iou"] <= 1.0
    assert metrics["fair:mean_iou"] < 1.0
