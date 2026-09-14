import numpy as np
from rasterio.transform import from_origin

from pl_hot.params import PostprocessParams
from pl_hot.postprocess import mask_to_feature_collection
from pl_hot.preprocess import PreprocessMeta


def test_mask_to_feature_collection_outputs_polygons() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    meta = PreprocessMeta(
        width=16,
        height=16,
        scale_x=1.0,
        scale_y=1.0,
        transform=from_origin(500000, 1000, 1, 1),
        crs="EPSG:3857",
    )
    fc = mask_to_feature_collection(
        mask,
        meta,
        PostprocessParams(mask_threshold=0.5, min_area_m2=1.0, hole_area_m2=1.0, simplify_m=0.0),
        source_name="chip.tif",
    )
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feature = fc["features"][0]
    assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    assert feature["properties"]["class_name"] == "parking"


def test_confidence_uses_polygon_interior_mean() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 1:3] = 1
    mask[7:9, 1:9] = 1  # L-shape with large bounding box
    prob = np.zeros((10, 10), dtype=np.float32)
    prob[mask == 1] = 0.9
    meta = PreprocessMeta(
        width=10,
        height=10,
        scale_x=1.0,
        scale_y=1.0,
        transform=from_origin(0, 10, 1, 1),
        crs="EPSG:3857",
    )
    fc = mask_to_feature_collection(
        mask,
        meta,
        PostprocessParams(mask_threshold=0.5, min_area_m2=1.0, hole_area_m2=1.0, simplify_m=0.0),
        probability=prob,
    )
    conf = fc["features"][0]["properties"]["confidence"]
    assert 0.85 <= conf <= 0.95
