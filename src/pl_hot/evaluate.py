"""Evaluation helpers for parking-lot segmentation."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decode import decode_segformer_onnx_output
from .params import PreprocessParams
from .preprocess import preprocess_chip_for_onnx


def _binary_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    return 1.0 if union == 0.0 else inter / union


def _class_ious(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    pred_fg = pred > 0
    truth_fg = truth > 0
    fg = _binary_iou(pred_fg, truth_fg)
    bg = _binary_iou(~pred_fg, ~truth_fg)
    return bg, fg


def evaluate_binary_masks(pred_masks_dir: str | Path, gt_masks_dir: str | Path) -> dict[str, float]:
    pred_dir = Path(pred_masks_dir)
    gt_dir = Path(gt_masks_dir)
    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        raise ValueError(f"No ground-truth masks found in {gt_dir}")

    miou_per_chip: list[float] = []
    accs: list[float] = []
    for gt in gt_files:
        pred = pred_dir / gt.name
        if not pred.exists():
            continue
        gt_arr = np.asarray(Image.open(gt).convert("L")) > 0
        pr_arr = np.asarray(Image.open(pred).convert("L")) > 0
        bg_iou, fg_iou = _class_ious(pr_arr, gt_arr)
        miou_per_chip.append((bg_iou + fg_iou) / 2.0)
        accs.append(float((pr_arr == gt_arr).mean()))

    if not miou_per_chip:
        raise ValueError("No overlapping mask names between prediction and ground-truth dirs")

    return {
        "fair:accuracy": float(np.mean(accs)),
        "fair:mean_iou": float(np.mean(miou_per_chip)),
    }


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pl-hot with `[train]` extras for model-driven evaluation.") from exc
    return torch, F


def _parking_logits_from_model_output(output: Any) -> np.ndarray:
    logits = output.logits if hasattr(output, "logits") else output
    logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
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
    """Run model inference on chips and compute fAIr PW/mIoU metrics."""
    torch, F = _require_torch()
    model = model.eval()
    pp = preprocess_cfg or PreprocessParams()
    chips = sorted(list(Path(images_dir).glob("*.tif")) + list(Path(images_dir).glob("*.tiff")))
    if not chips:
        raise ValueError(f"No chips found in {images_dir}")

    preds_tmp = Path(masks_dir).parent / "_pred_masks_tmp"
    preds_tmp.mkdir(parents=True, exist_ok=True)
    produced = 0
    with torch.no_grad():
        for chip in chips:
            gt_path = Path(masks_dir) / f"{chip.stem}.png"
            if not gt_path.exists():
                continue
            x_np, _ = preprocess_chip_for_onnx(chip, pp)
            x = torch.tensor(x_np, dtype=torch.float32)
            out = model(pixel_values=x)
            logits = out.logits if hasattr(out, "logits") else out
            gt_arr = np.asarray(Image.open(gt_path).convert("L"))
            logits = F.interpolate(logits, size=gt_arr.shape, mode="bilinear", align_corners=False)
            park_logits = _parking_logits_from_model_output(logits)
            mask, _ = decode_segformer_onnx_output(park_logits, threshold=threshold)
            Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(preds_tmp / gt_path.name)
            produced += 1
    if produced == 0:
        raise ValueError("No predictions produced: ensure mask names match chip stems.")
    return evaluate_binary_masks(preds_tmp, masks_dir)
