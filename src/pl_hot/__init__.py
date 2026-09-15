"""pl-hot: reusable parking-lot segmentation helpers for fAIr."""

from .checkpoint import (
    HF_SEGFORMER_B5,
    encoder_variant_from_keys,
    inspect_checkpoint,
    load_segformer_from_checkpoint,
)
from .dataset import prepare_seg_dataset_from_geojson, spatial_split
from .decode import decode_segformer_onnx_output
from .evaluate import evaluate_binary_masks, evaluate_segformer
from .export import export_onnx_bytes
from .params import (
    InferenceParams,
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    TrainParams,
    parse_inference_params,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)
from .postprocess import mask_to_feature_collection
from .preprocess import PreprocessMeta, preprocess_chip_for_onnx
from .serve import iter_image_paths, predict_session
from .train import train_segformer

__all__ = [
    "InferenceParams",
    "PostprocessParams",
    "PreprocessMeta",
    "PreprocessParams",
    "SplitParams",
    "TrainParams",
    "HF_SEGFORMER_B5",
    "decode_segformer_onnx_output",
    "encoder_variant_from_keys",
    "evaluate_binary_masks",
    "evaluate_segformer",
    "export_onnx_bytes",
    "inspect_checkpoint",
    "load_segformer_from_checkpoint",
    "iter_image_paths",
    "mask_to_feature_collection",
    "parse_inference_params",
    "parse_postprocess_params",
    "parse_preprocess_params",
    "parse_split_params",
    "parse_train_params",
    "predict_session",
    "prepare_seg_dataset_from_geojson",
    "preprocess_chip_for_onnx",
    "spatial_split",
    "train_segformer",
]
