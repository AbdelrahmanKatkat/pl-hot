"""ONNX output decoding for binary parking-lot segmentation."""

import numpy as np
from PIL import Image


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bilinear_logits(logits: np.ndarray, size: int) -> np.ndarray:
    """Resize (1, C, H, W) logits to `size`×`size` before sigmoid/softmax."""
    _, channels, height, width = logits.shape
    if height == size and width == size:
        return logits
    out = np.empty((1, channels, size, size), dtype=np.float32)
    for channel in range(channels):
        out[0, channel] = np.asarray(
            Image.fromarray(logits[0, channel].astype(np.float32), mode="F").resize((size, size), Image.BILINEAR),
            dtype=np.float32,
        )
    return out


def decode_segformer_onnx_output(
    output: np.ndarray,
    threshold: float = 0.5,
    spatial_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ONNX tensor to (binary_mask, parking_probability).

    Supported shapes:
    - (1, 1, H, W): BCE logits
    - (1, 2, H, W): two-class logits (background, parking)

    Hugging Face SegFormer often emits H/4 × W/4 logits. Pass `spatial_size`
    (`model_input_size`) so bilinear upsample happens on logits, then threshold —
    same order as training `F.interpolate` on logits, not nearest on a tiny mask.
    """
    logits = np.asarray(output, dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"Expected output shape (1,C,H,W), got {logits.shape}")

    if spatial_size is not None:
        logits = _bilinear_logits(logits, int(spatial_size))

    channels = logits.shape[1]
    if channels == 1:
        parking_prob = _sigmoid(logits[0, 0])
    elif channels == 2:
        exp_logits = np.exp(logits[0] - np.max(logits[0], axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        parking_prob = probs[1]
    else:
        raise ValueError(f"Expected 1 or 2 channels, got {channels}")

    mask = (parking_prob >= float(threshold)).astype(np.uint8)
    return mask, parking_prob.astype(np.float32)
