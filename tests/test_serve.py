import numpy as np

from pl_hot.serve import predict_session


class _Input:
    name = "image"


class FakeSession:
    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feeds):
        batch = feeds["image"]
        _, _, h, w = batch.shape
        logits = np.full((1, 1, h, w), -5.0, dtype=np.float32)
        logits[:, :, h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] = 5.0
        return [logits]


def test_predict_session_returns_geojson(chips_dir) -> None:
    result = predict_session(FakeSession(), chips_dir, {"model_input_size": 32, "min_area_m2": 1.0})
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) >= 1


class QuarterResSession:
    """HF SegFormer-style logits at H/4 × W/4 of the preprocessed tensor."""

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feeds):
        batch = feeds["image"]
        _, _, h, w = batch.shape
        qh, qw = h // 4, w // 4
        logits = np.full((1, 1, qh, qw), -5.0, dtype=np.float32)
        logits[:, :, qh // 4 : (3 * qh) // 4, qw // 4 : (3 * qw) // 4] = 5.0
        return [logits]


def test_predict_session_upsamples_quarter_res_logits(chips_dir) -> None:
    result = predict_session(QuarterResSession(), chips_dir, {"model_input_size": 32, "min_area_m2": 1.0})
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) >= 1
