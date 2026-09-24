"""Read a Lightning/HF SegFormer `.ckpt` with `torch.load`.

A `.ckpt` is not named `.zip`, but PyTorch Lightning saves it as a zip of pickle
+ tensor shards. Callers should use `torch.load`, not unzip it by hand.

`inspect_checkpoint` returns what is in the file (encoder from key depths,
channel counts from tensor shapes). Tile size 512 and ImageNet norm are not
stored in the weight blob; those live in `HF_SEGFORMER_B5` / train defaults.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Config source for MiT-B5 SegFormer. Note: this does **not** pin input size; we still
# train/export at 512 by default. We only need a correct architecture config.
#
# We vendor a minimal config JSON in `pl_hot/assets/` to avoid network access in
# common workflows (exporting a local `.ckpt` to ONNX).
HF_SEGFORMER_B5 = "nvidia/segformer-b5-finetuned-ade-640-640"
_BUNDLED_B5_CONFIG = "segformer_b5_ade_640_config.min.json"

_BLOCK_RE = re.compile(r"(?:^|\.)(?:model\.)?segformer\.encoder\.block\.(\d+)\.(\d+)\.")
_PATCH_PROJ = "segformer.encoder.patch_embeddings.0.proj.weight"
_CLASSIFIER = "decode_head.classifier.weight"

_DEPTHS_TO_VARIANT = {
    (2, 2, 2, 2): "mit-b0-or-b1",
    (3, 4, 6, 3): "mit-b2",
    (3, 4, 18, 3): "mit-b3",
    (3, 8, 27, 3): "mit-b4",
    (3, 6, 40, 3): "mit-b5",
}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pl-hot with `[train]` extras to read checkpoints.") from exc
    return torch


def _state_dict_from_checkpoint(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob
    raise ValueError("Unsupported checkpoint format")


def _strip_prefix(key: str, prefix: str = "model.") -> str:
    return key[len(prefix) :] if key.startswith(prefix) else key


def encoder_variant_from_keys(keys: list[str]) -> str | None:
    """Infer MiT variant from `segformer.encoder.block.{stage}.{idx}` keys."""
    stages: dict[int, int] = {}
    for key in keys:
        match = _BLOCK_RE.search(key)
        if match is None:
            continue
        stage, idx = int(match.group(1)), int(match.group(2))
        stages[stage] = max(stages.get(stage, -1), idx)
    if not stages:
        joined = " ".join(keys).lower()
        if "segformer.encoder" in joined or "segformer" in joined:
            return "segformer-unknown-variant"
        return None
    depths = tuple(stages[i] + 1 for i in range(max(stages) + 1))
    return _DEPTHS_TO_VARIANT.get(depths, f"mit-unknown-depths-{depths}")


def _tensor_shape(state: dict[str, Any], suffix: str) -> tuple[int, ...] | None:
    """Return `.shape` of the first tensor whose key ends with `suffix`."""
    for key, value in state.items():
        if key.endswith(suffix) or _strip_prefix(key).endswith(suffix):
            if hasattr(value, "shape"):
                return tuple(int(d) for d in value.shape)
    return None


def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load the ckpt and return encoder/channel facts from its tensors."""
    torch = _require_torch()
    path = Path(checkpoint_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = _state_dict_from_checkpoint(blob)
    in_shape = _tensor_shape(state, _PATCH_PROJ)
    out_shape = _tensor_shape(state, _CLASSIFIER)
    return {
        "checkpoint_path": str(path.resolve()),
        "encoder_variant": encoder_variant_from_keys(list(state.keys())),
        "input_channels": int(in_shape[1]) if in_shape and len(in_shape) >= 2 else None,
        "output_channels": int(out_shape[0]) if out_shape else None,
        "patch_embed_shape": in_shape,
        "classifier_shape": out_shape,
    }


def load_segformer_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    hf_pretrained: str = HF_SEGFORMER_B5,
    num_labels: int = 2,
) -> Any:
    """Build HF SegFormer-B5 and load Lightning `model.*` weights."""
    torch = _require_torch()
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pl-hot with `[train]` extras to load SegFormer weights.") from exc

    # Avoid downloading the full HF weight blob.
    # Prefer the bundled config to avoid network access. Fall back to HF if the
    # package data is missing or the caller overrode `hf_pretrained`.
    cfg = None
    try:
        import json
        import importlib.resources as ir

        raw = ir.files("pl_hot.assets").joinpath(_BUNDLED_B5_CONFIG).read_text(encoding="utf-8")
        cfg = SegformerConfig.from_dict(json.loads(raw))
    except Exception:
        cfg = None
    if cfg is None:
        cfg = SegformerConfig.from_pretrained(hf_pretrained)
    cfg.num_labels = int(num_labels)
    model = SegformerForSemanticSegmentation(cfg)
    blob = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = _state_dict_from_checkpoint(blob)
    stripped = {_strip_prefix(k): v for k, v in state.items() if not k.startswith("optimizer")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    model.eval()
    model._pl_hot_load = {"missing": list(missing), "unexpected": list(unexpected)}
    return model
