"""ONNX output decoding for binary parking-lot segmentation."""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_segformer_onnx_output(output: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Decode ONNX tensor to (binary_mask, parking_probability).

    Supported shapes:
    - (1, 1, H, W): BCE logits
    - (1, 2, H, W): two-class logits (background, parking)
    """
    logits = np.asarray(output)
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"Expected output shape (1,C,H,W), got {logits.shape}")

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
