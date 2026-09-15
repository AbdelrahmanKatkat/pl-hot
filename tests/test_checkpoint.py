from pl_hot.checkpoint import HF_SEGFORMER_B5, _tensor_shape, encoder_variant_from_keys


class _T:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_encoder_variant_from_b5_keys() -> None:
    keys = []
    for stage, depth in ((0, 3), (1, 6), (2, 40), (3, 3)):
        for idx in range(depth):
            keys.append(f"model.segformer.encoder.block.{stage}.{idx}.layer_norm_1.weight")
    assert encoder_variant_from_keys(keys) == "mit-b5"
    assert HF_SEGFORMER_B5.endswith("512-512")


def test_tensor_shape_strips_lightning_prefix() -> None:
    state = {
        "model.segformer.encoder.patch_embeddings.0.proj.weight": _T((64, 3, 7, 7)),
        "model.decode_head.classifier.weight": _T((2, 768, 1, 1)),
    }
    assert _tensor_shape(state, "segformer.encoder.patch_embeddings.0.proj.weight") == (64, 3, 7, 7)
    assert _tensor_shape(state, "decode_head.classifier.weight") == (2, 768, 1, 1)
