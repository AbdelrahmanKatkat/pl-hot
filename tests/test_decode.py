import numpy as np

from pl_hot.decode import decode_segformer_onnx_output


def test_decode_single_channel_logits() -> None:
    logits = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
    logits[0, 0, 1:3, 1:3] = 10.0
    mask, prob = decode_segformer_onnx_output(logits, threshold=0.5)
    assert mask.shape == (4, 4)
    assert prob.shape == (4, 4)
    assert mask.sum() == 4


def test_decode_two_channel_logits() -> None:
    logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    logits[0, 1, :, :] = 3.0
    mask, _ = decode_segformer_onnx_output(logits, threshold=0.5)
    assert int(mask.sum()) == 4
