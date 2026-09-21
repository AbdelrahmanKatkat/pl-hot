"""Centralized parameters for `pl-hot` (defaults + parsing).

Design goals:
- One place for training / preprocess / inference / postprocess knobs.
- Accept plain dicts from fAIr `pipeline.py` and return typed dataclasses.

Inference vs postprocess (parking, from the WACV 2025 paper):
- Inference = model-space decode: sigmoid/softmax vs `mask_threshold` (0.5).
- Postprocess = map-space cleanup: 60 m² holes, Douglas–Peucker, min area.
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SplitParams:
    """Paper train/val split is 90/10. OAM chips use the same spatial block split algorithm."""

    val_ratio: float = 0.1
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class TrainParams:
    """Paper: Adam, lr=1e-5, BCE-with-logits, pos_weight≈4.76, early-stop 10."""

    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    pos_weight: float = 4.76
    early_stop_patience: int = 10
    freeze_encoder: bool = True
    sample_fraction: float = 1.0
    model_input_size: int = 512
    device: str = "auto"  # cuda if visible, else cpu


@dataclass(frozen=True)
class PreprocessParams:
    """RGB chip → NCHW. Paper tiles are 512×512 with ImageNet mean/std."""

    model_input_size: int = 512
    chip_size_hint: int = 256
    normalize_01: bool = True
    imagenet_norm: bool = True


@dataclass(frozen=True)
class InferenceParams:
    """User-facing decode knobs for fAIr `predict(..., params=...)`.

    STAC often names this `inference.confidence_threshold`; we also accept
    `mask_threshold` (paper BCE-with-logits, default 0.5).
    """

    mask_threshold: float = 0.5


@dataclass(frozen=True)
class PostprocessParams:
    """Polygon cleanup after decode (WACV 2025 §4.2). Not decode."""

    min_area_m2: float = 60.0
    hole_area_m2: float = 60.0
    simplify_m: float = 1.0


def _get(d: Mapping[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d and d[key] is not None else default


def _as_bool(v: Any) -> bool:
    """STAC/JSON often sends `"true"` / `"0"`. `bool("false")` is True in Python."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        raw = v.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got {v!r}")


def _as_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError("bool is not a valid int here")
    return int(v)


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        raise ValueError("bool is not a valid float here")
    return float(v)


def parse_split_params(hyperparameters: Mapping[str, Any] | None) -> SplitParams:
    hp = hyperparameters or {}
    return SplitParams(
        val_ratio=_as_float(_get(hp, "val_ratio", SplitParams.val_ratio)),
        split_seed=_as_int(_get(hp, "split_seed", SplitParams.split_seed)),
        block_size=_as_int(_get(hp, "block_size", SplitParams.block_size)),
    )


def parse_train_params(hyperparameters: Mapping[str, Any] | None) -> TrainParams:
    hp = hyperparameters or {}
    return TrainParams(
        epochs=_as_int(_get(hp, "epochs", TrainParams.epochs)),
        batch_size=_as_int(_get(hp, "batch_size", TrainParams.batch_size)),
        learning_rate=_as_float(_get(hp, "learning_rate", TrainParams.learning_rate)),
        weight_decay=_as_float(_get(hp, "weight_decay", TrainParams.weight_decay)),
        pos_weight=_as_float(_get(hp, "pos_weight", TrainParams.pos_weight)),
        early_stop_patience=_as_int(_get(hp, "early_stop_patience", TrainParams.early_stop_patience)),
        freeze_encoder=_as_bool(_get(hp, "freeze_encoder", TrainParams.freeze_encoder)),
        sample_fraction=_as_float(_get(hp, "sample_fraction", TrainParams.sample_fraction)),
        model_input_size=_as_int(_get(hp, "chip_size", _get(hp, "model_input_size", TrainParams.model_input_size))),
        device=str(_get(hp, "device", TrainParams.device)),
    )


def parse_preprocess_params(hyperparameters: Mapping[str, Any] | None) -> PreprocessParams:
    hp = hyperparameters or {}
    size = _get(hp, "model_input_size", None)
    if size is None:
        size = _get(hp, "chip_size", PreprocessParams.model_input_size)
    return PreprocessParams(
        model_input_size=_as_int(size),
        chip_size_hint=_as_int(_get(hp, "chip_size_hint", PreprocessParams.chip_size_hint)),
        normalize_01=_as_bool(_get(hp, "normalize_01", PreprocessParams.normalize_01)),
        imagenet_norm=_as_bool(_get(hp, "imagenet_norm", PreprocessParams.imagenet_norm)),
    )


def parse_inference_params(params: Mapping[str, Any] | None) -> InferenceParams:
    """Parse decode knobs from fAIr `predict` / `run_inference` params."""
    p = params or {}
    threshold = _get(p, "mask_threshold", None)
    if threshold is None:
        threshold = _get(p, "confidence_threshold", InferenceParams.mask_threshold)
    return InferenceParams(mask_threshold=_as_float(threshold))


def parse_postprocess_params(params: Mapping[str, Any] | None) -> PostprocessParams:
    p = params or {}
    return PostprocessParams(
        min_area_m2=_as_float(_get(p, "min_area_m2", PostprocessParams.min_area_m2)),
        hole_area_m2=_as_float(_get(p, "hole_area_m2", PostprocessParams.hole_area_m2)),
        simplify_m=_as_float(_get(p, "simplify_m", PostprocessParams.simplify_m)),
    )
