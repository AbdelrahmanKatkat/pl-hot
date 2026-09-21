"""Evaluation helpers for parking-lot segmentation.

Metrics follow the WACV 2025 paper names (PW, mIoU) with dataset-level
(micro) aggregation used by MMSegmentation / Cityscapes, not a mean of
per-chip scores.
"""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import list_chip_paths
from .decode import decode_segformer_onnx_output
from .params import PreprocessParams
from .preprocess import preprocess_chip_for_onnx


def _confusion(pred_fg: np.ndarray, truth_fg: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, tn, fp, fn) for the parking (foreground) class."""
    pred_fg = pred_fg.astype(bool)
    truth_fg = truth_fg.astype(bool)
    tp = int(np.logical_and(pred_fg, truth_fg).sum())
    tn = int(np.logical_and(~pred_fg, ~truth_fg).sum())
    fp = int(np.logical_and(pred_fg, ~truth_fg).sum())
    fn = int(np.logical_and(~pred_fg, truth_fg).sum())
    return tp, tn, fp, fn


def _iou(intersect: int, union: int) -> float:
    if union == 0:
        return float("nan")
    return intersect / union


def _metrics_from_confusion(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    n = tp + tn + fp + fn
    if n == 0:
        raise ValueError("No pixels to evaluate")
    iou_parking = _iou(tp, tp + fp + fn)
    iou_background = _iou(tn, tn + fp + fn)
    class_ious = [v for v in (iou_background, iou_parking) if not np.isnan(v)]
    if not class_ious:
        raise ValueError("All class unions are empty")
    out: dict[str, float] = {
        "accuracy": (tp + tn) / n,
        "mean_iou": float(np.mean(class_ious)),
    }
    if not np.isnan(iou_background):
        out["iou_background"] = iou_background
    if not np.isnan(iou_parking):
        out["iou_parking_lot"] = iou_parking
    return out


def evaluate_binary_masks(pred_masks_dir: str | Path, gt_masks_dir: str | Path) -> dict[str, float]:
    """Dataset-level PW (aAcc) and 2-class mIoU from PNG mask folders."""
    pred_dir = Path(pred_masks_dir)
    gt_dir = Path(gt_masks_dir)
    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        raise ValueError(f"No ground-truth masks found in {gt_dir}")

    tp = tn = fp = fn = 0
    used = 0
    for gt in gt_files:
        pred = pred_dir / gt.name # same name as the ground-truth mask
        if not pred.exists():
            continue
        gt_arr = np.asarray(Image.open(gt).convert("L")) > 0
        pr_arr = np.asarray(Image.open(pred).convert("L")) > 0
        if pr_arr.shape != gt_arr.shape:
            raise ValueError(f"Shape mismatch for {gt.name}: pred {pr_arr.shape} vs gt {gt_arr.shape}")
        ctp, ctn, cfp, cfn = _confusion(pr_arr, gt_arr)
        tp += ctp
        tn += ctn
        fp += cfp
        fn += cfn
        used += 1

    if used == 0:
        raise ValueError("No overlapping mask names between prediction and ground-truth dirs")

    return _metrics_from_confusion(tp, tn, fp, fn)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pl-hot with `[train]` extras for model-driven evaluation.") from exc
    return torch, F


def _parking_logits_from_model_output(output: Any) -> np.ndarray:
    """Return (1, 1, H, W) parking logits. ℓ_park − ℓ_bg ≡ softmax parking after sigmoid."""
    logits = output.logits if hasattr(output, "logits") else output
    logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    if logits_np.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W) logits, got {logits_np.shape}")
    if logits_np.shape[1] == 1:
        return logits_np
    if logits_np.shape[1] == 2:
        return (logits_np[:, 1:2] - logits_np[:, 0:1]).astype(np.float32)
    raise ValueError(f"Expected 1 or 2 channels, got {logits_np.shape[1]}")


def evaluate_segformer(
    model: Any,
    *,
    images_dir: str | Path,
    masks_dir: str | Path,
    threshold: float = 0.5,
    preprocess_cfg: PreprocessParams | None = None,
) -> dict[str, float]:
    """Run the torch model on chips and score **raw** pixel masks (no polygon postprocess).

    Logits are bilinear-upsampled to `model_input_size` (train grid), thresholded,
    then nearest-unscaled to the native GT mask — the same two resizes as serve.
    """
    torch, F = _require_torch()
    model = model.eval()
    pp = preprocess_cfg or PreprocessParams()
    chips = list_chip_paths(images_dir)
    if not chips:
        raise ValueError(f"No chips found in {images_dir}")

    preds_tmp = Path(masks_dir).parent / "_pred_masks_tmp"
    preds_tmp.mkdir(parents=True, exist_ok=True)
    produced = 0
    model_hw = (int(pp.model_input_size), int(pp.model_input_size))
    with torch.no_grad():
        for chip in chips:
            gt_path = Path(masks_dir) / f"{chip.stem}.png"
            if not gt_path.exists():
                continue
            x_np, _ = preprocess_chip_for_onnx(chip, pp)
            x = torch.tensor(x_np, dtype=torch.float32)
            out = model(pixel_values=x)
            logits = out.logits if hasattr(out, "logits") else out
            # Same grid as train loss: bilinear logits → model_input_size, then threshold.
            logits = F.interpolate(logits, size=model_hw, mode="bilinear", align_corners=False)
            park_logits = _parking_logits_from_model_output(logits)
            mask, _ = decode_segformer_onnx_output(park_logits, threshold=threshold)
            gt_arr = np.asarray(Image.open(gt_path).convert("L"))
            gt_h, gt_w = gt_arr.shape[:2]
            # Same unscale as serve: nearest model-space mask → native chip (map space).
            if mask.shape != (gt_h, gt_w):
                mask = np.asarray(
                    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").resize((gt_w, gt_h), Image.NEAREST)
                )
            Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(preds_tmp / gt_path.name)
            produced += 1
    if produced == 0:
        raise ValueError("No predictions produced: ensure mask names match chip stems.")
    return evaluate_binary_masks(preds_tmp, masks_dir)
