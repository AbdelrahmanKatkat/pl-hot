"""Training helper for SegFormer parking-lot finetuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import HF_SEGFORMER_B5, load_segformer_from_checkpoint
from .dataset import list_chip_paths
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
        self.images = list_chip_paths(self.images_dir)
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
        # Labels are 0/1, not RGB: no /255, no ImageNet. Resize with nearest so class ids stay 0 or 1.
        size = int(self.preprocess_cfg.model_input_size)
        mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
        y = (np.asarray(mask) > 0).astype(np.float32)
        if x.shape[-2:] != y.shape:
            raise ValueError(
                f"Image/mask spatial size mismatch after preprocess: image {tuple(x.shape)} vs mask {tuple(y.shape)}"
            )
        return x, y


def _resolve_device(requested: str, torch: Any) -> str:
    """Use CUDA when visible unless the caller forced CPU."""
    raw = (requested or "auto").strip().lower()
    if raw in {"cpu", "cpu:0"}:
        return "cpu"
    if torch.cuda.is_available():
        if raw in {"auto", "cuda", "gpu", "0"}:
            return "cuda"
        return requested
    return "cpu"


def _extract_logits(logits: Any) -> Any:
    tensor = logits.logits if hasattr(logits, "logits") else logits
    if tensor.ndim != 4:
        raise ValueError(f"Expected logits with shape (B,C,H,W), got {tuple(tensor.shape)}")
    if tensor.shape[1] == 1:
        return tensor
    if tensor.shape[1] == 2:
        return (tensor[:, 1:2, :, :] - tensor[:, 0:1, :, :]).contiguous()
    raise ValueError(f"Expected 1 or 2 output channels, got {tensor.shape[1]}")


def _batch_loss(model: Any, x_np: Any, y_np: Any, *, device: str, torch: Any, F: Any, criterion: Any) -> Any:
    x = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_np), dtype=torch.float32, device=device).unsqueeze(1)
    logits = _extract_logits(model(pixel_values=x))
    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return criterion(logits, y)


def _run_eval(model: Any, loader: Any, device: str, torch: Any, F: Any, criterion: Any) -> float:
    """Mean val loss. No backward; weights do not change."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x_np, y_np in loader:
            loss = _batch_loss(model, x_np, y_np, device=device, torch=torch, F=F, criterion=criterion)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _run_train(model: Any, loader: Any, device: str, torch: Any, F: Any, criterion: Any, optimizer: Any) -> float:
    """One epoch: forward, backward, Adam step. Returns mean train loss."""
    model.train()
    losses: list[float] = []
    for x_np, y_np in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, x_np, y_np, device=device, torch=torch, F=F, criterion=criterion)
        loss.backward()
        optimizer.step()
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
    device = _resolve_device(cfg.device, torch)
    # ImageNet mean/std are defined on [0, 1], so /255 then (x-mean)/std. Both flags on is HF SegFormer.
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
        train_loss = _run_train(model, train_loader, device, torch, F, criterion, optimizer)
        history["train_loss"].append(train_loss)

        val_loss = train_loss
        if val_loader is not None:
            val_loss = _run_eval(model, val_loader, device, torch, F, criterion)
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
