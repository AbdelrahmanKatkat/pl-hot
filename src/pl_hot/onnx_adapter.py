"""HF SegFormer → tensor-in / tensor-out for `torch.onnx.export`.

Imported only from `export.py` (needs `[train]` extras). Not used at serve time.
"""

from typing import Any

import torch


class LogitsOnly(torch.nn.Module):
    """Wrap a trained SegFormer so ONNX sees one image tensor and one logits tensor.

    Hugging Face `forward` takes `pixel_values=` and returns a dataclass.
    `wrapped` is that model; this module does not add weights.
    """

    def __init__(self, wrapped: Any) -> None:
        super().__init__()
        # The trained HF SegFormer (encoder + decode head). No extra weights here.
        self.wrapped = wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. ONNX feeds a positional image tensor `x` (1, 3, 512, 512).
        #    HF SegFormer expects that tensor as the keyword `pixel_values`.
        try:
            out = self.wrapped(pixel_values=x)
        except TypeError:
            # Plain nn.Module that already takes `forward(x)`.
            out = self.wrapped(x)
        # 2. HF returns a dataclass (logits, hidden states, …). ONNX cannot
        #    record that object — keep only the logits tensor (1, 2, H, W).
        if hasattr(out, "logits"):
            return out.logits
        return out
