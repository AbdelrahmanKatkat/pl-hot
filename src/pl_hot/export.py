"""ONNX export helpers for SegFormer models."""

import tempfile
from pathlib import Path
from typing import Any


def export_onnx_bytes(model: Any, model_input_size: int = 512) -> bytes:
    """Export a torch segmentation model to self-contained ONNX bytes."""
    try:
        import onnx
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError("Install pl-hot with `[train]` extras for ONNX export support.") from exc

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            try:
                out = self.wrapped(pixel_values=x)
            except TypeError:
                out = self.wrapped(x)
            return out.logits if hasattr(out, "logits") else out

    model = _LogitsOnly(model).cpu().eval()
    dummy = torch.randn(1, 3, model_input_size, model_input_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            dynamo=True,
        )
        onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
        onnx.checker.check_model(str(onnx_path))
        return onnx_path.read_bytes()
