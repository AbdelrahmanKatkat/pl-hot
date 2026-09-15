from pl_hot.params import parse_inference_params, parse_postprocess_params


def test_inference_threshold_is_separate_from_postprocess() -> None:
    inf = parse_inference_params({"confidence_threshold": "0.7"})
    post = parse_postprocess_params({"min_area_m2": "30"})
    assert inf.mask_threshold == 0.7
    assert post.min_area_m2 == 30.0
    assert not hasattr(post, "mask_threshold")
