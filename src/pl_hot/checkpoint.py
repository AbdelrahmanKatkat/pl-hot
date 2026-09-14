"""Checkpoint inspection helpers for SegFormer contract validation."""

import json
from pathlib import Path
from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pl-hot with `[train]` extras for checkpoint inspection.") from exc
    return torch


def _state_dict_from_checkpoint(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob
    raise ValueError("Unsupported checkpoint format")


def _infer_encoder_variant(keys: list[str]) -> str | None:
    joined = " ".join(keys).lower()
    for name in ("mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"):
        if name in joined:
            return name
    if "segformer.encoder" in joined or "segformer" in joined:
        return "segformer-unknown-variant"
    return None


def _infer_input_channels(state: dict[str, Any]) -> int | None:
    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        shape = tuple(value.shape)
        if len(shape) == 4 and "weight" in key:
            return int(shape[1])
    return None


def _infer_output_channels(state: dict[str, Any]) -> int | None:
    candidates = [k for k in state if k.endswith("classifier.weight") or "decode_head.classifier.weight" in k]
    for key in candidates:
        value = state[key]
        if hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
    return None


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def inspect_checkpoint_contract(
    checkpoint_path: str | Path,
    *,
    preprocessor_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect the five README facts needed before freezing ONNX shapes."""
    torch = _require_torch()
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    state = _state_dict_from_checkpoint(ckpt)
    keys = list(state.keys())

    model_cfg = _load_json(model_config_path)
    pre_cfg = _load_json(preprocessor_config_path)

    input_size = None
    if isinstance(pre_cfg, dict):
        size = pre_cfg.get("size")
        if isinstance(size, dict):
            input_size = size.get("height") or size.get("width")
        elif isinstance(size, int):
            input_size = size

    image_mean = pre_cfg.get("image_mean") if isinstance(pre_cfg, dict) else None
    image_std = pre_cfg.get("image_std") if isinstance(pre_cfg, dict) else None
    normalization = "imagenet" if image_mean and image_std else "unknown"

    facts = {
        "checkpoint_path": str(checkpoint_path),
        "encoder_variant": _infer_encoder_variant(keys),
        "model_input_size": input_size,
        "output_channels": _infer_output_channels(state),
        "input_channels": _infer_input_channels(state),
        "normalization": normalization,
        "image_mean": image_mean,
        "image_std": image_std,
    }

    warnings: list[str] = []
    if facts["input_channels"] not in {3, None}:
        warnings.append(f"Checkpoint expects {facts['input_channels']} input channels (fAIr requires RGB=3).")
    if facts["output_channels"] not in {1, 2, None}:
        warnings.append(f"Unexpected output channels: {facts['output_channels']}; expected 1 or 2.")
    if facts["model_input_size"] is None:
        warnings.append("Input size not found; provide `preprocessor_config_path` to lock ONNX shape.")
    if facts["normalization"] == "unknown":
        warnings.append("Normalization unknown; provide preprocessor config or training metadata.")
    facts["warnings"] = warnings
    return facts
