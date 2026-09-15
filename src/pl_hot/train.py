"""Training helper for SegFormer parking-lot finetuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import HF_SEGFORMER_B5, load_segformer_from_checkpoint
from .params import PreprocessParams, TrainParams
from .preprocess import preprocess_chip_for_onnx


def _require_train_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError("Install pl-hot with `[train]` extras for training support.") from exc
    return torch, F, SegformerForSemanticSegmentation


@dataclass(frozen=True)
class TrainResult:
    model: Any
    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float


class _SegDataset:
    def __init__(self, images_dir: str | Path, masks_dir: str | Path, preprocess_cfg: PreprocessParams) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.preprocess_cfg = preprocess_cfg
        self.images = sorted(list(self.images_dir.glob("*.tif")) + list(self.images_dir.glob("*.tiff")))
        if not self.images:
            raise ValueError(f"No training chips found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        chip = self.images[idx]
        mask_path = self.masks_dir / f"{chip.stem}.png"
        if not mask_path.exists():
            raise ValueError(f"Missing mask for chip {chip.name}: {mask_path}")
        batch, _ = preprocess_chip_for_onnx(chip, self.preprocess_cfg)
        x = batch[0]
        y = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.float32)
        return x, y


def _extract_logits(logits: Any) -> Any:
    tensor = logits.logits if hasattr(logits, "logits") else logits
    if tensor.ndim != 4:
        raise ValueError(f"Expected logits with shape (B,C,H,W), got {tuple(tensor.shape)}")
    if tensor.shape[1] == 1:
        return tensor
    if tensor.shape[1] == 2:
        return (tensor[:, 1:2, :, :] - tensor[:, 0:1, :, :]).contiguous()
    raise ValueError(f"Expected 1 or 2 output channels, got {tensor.shape[1]}")


def _run_eval(model: Any, loader: Any, device: str, torch: Any, F: Any, pos_weight: float) -> float:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    losses: list[float] = []
    with torch.no_grad():
        for x_np, y_np in loader:
            x = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=device)
            y = torch.tensor(np.asarray(y_np), dtype=torch.float32, device=device).unsqueeze(1)
            logits = _extract_logits(model(pixel_values=x))
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, y)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_segformer(
    *,
    train_images_dir: str | Path,
    train_masks_dir: str | Path,
    cfg: TrainParams,
    val_images_dir: str | Path | None = None,
    val_masks_dir: str | Path | None = None,
    pretrained_model_name_or_path: str = HF_SEGFORMER_B5,
    checkpoint_path: str | Path | None = None,
) -> TrainResult:
    """Fine-tune SegFormer-B5 on chip/mask pairs and return best model + history.

    Pass `checkpoint_path` to start from the published Lightning `SegFormer_large_parking.ckpt`.
    """
    torch, F, SegformerForSemanticSegmentation = _require_train_deps()
    device = "cpu" if cfg.device in {"cpu", ""} else cfg.device
    preprocess_cfg = PreprocessParams(model_input_size=cfg.model_input_size, normalize_01=True, imagenet_norm=True)
    train_ds = _SegDataset(train_images_dir, train_masks_dir, preprocess_cfg)
    sample_count = len(train_ds)
    if 0 < cfg.sample_fraction < 1.0:
        sample_count = max(1, int(round(len(train_ds) * cfg.sample_fraction)))
        indices = torch.randperm(len(train_ds))[:sample_count].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    val_loader = None
    if val_images_dir is not None and val_masks_dir is not None:
        val_ds = _SegDataset(val_images_dir, val_masks_dir, preprocess_cfg)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if checkpoint_path is not None:
        model = load_segformer_from_checkpoint(checkpoint_path, hf_pretrained=pretrained_model_name_or_path)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
    if cfg.freeze_encoder and hasattr(model, "segformer"):
        for p in model.segformer.parameters():
            p.requires_grad = False
    model = model.to(device)
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = float("inf")
    patience_left = cfg.early_stop_patience

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses: list[float] = []
        for x_np, y_np in train_loader:
            x = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=device)
            y = torch.tensor(np.asarray(y_np), dtype=torch.float32, device=device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = _extract_logits(model(pixel_values=x))
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        history["train_loss"].append(train_loss)

        val_loss = train_loss
        if val_loader is not None:
            val_loss = _run_eval(model, val_loader, device, torch, F, cfg.pos_weight)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    if sample_count < len(getattr(train_ds, "dataset", train_ds)):
        history["sample_count"] = [float(sample_count)]

    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
    )
