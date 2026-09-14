from pl_hot.params import PreprocessParams
from pl_hot.preprocess import preprocess_chip_for_onnx


def test_preprocess_tensor_shape_and_meta(chips_dir) -> None:
    chip = next(chips_dir.glob("*.tif"))
    batch, meta = preprocess_chip_for_onnx(chip, PreprocessParams(model_input_size=64))
    assert batch.shape == (1, 3, 64, 64)
    assert batch.dtype.name == "float32"
    assert meta.width == 32
    assert meta.height == 32
