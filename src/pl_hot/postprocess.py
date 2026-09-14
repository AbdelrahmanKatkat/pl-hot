"""Mask postprocessing: unscale, vectorize, clean, and emit GeoJSON."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image
from rasterio.features import rasterize, shapes
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .georef import area_m2, mapping_in_wgs84
from .params import PostprocessParams
from .preprocess import PreprocessMeta


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    as_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    resized = as_img.resize((width, height), Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def _iter_polygons(mask: np.ndarray, transform: Any) -> Iterable[Polygon]:
    for geom, value in shapes(mask.astype(np.uint8), mask=(mask > 0), transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if isinstance(poly, Polygon):
            yield poly
        elif isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                yield p


def _drop_small_holes(poly: Polygon, src_crs: Any, hole_area_m2: float) -> Polygon:
    if not poly.interiors:
        return poly
    keep = []
    for ring in poly.interiors:
        hole = Polygon(ring)
        if area_m2(hole, src_crs) >= hole_area_m2:
            keep.append(ring.coords)
    return Polygon(poly.exterior.coords, holes=keep)


def _simplify_geom(geom: BaseGeometry, src_crs: Any, tolerance_m: float) -> BaseGeometry:
    if tolerance_m <= 0:
        return geom
    # Simplify in source CRS; area thresholds already compensate for non-metric CRSs.
    return geom.simplify(tolerance_m, preserve_topology=True)


def mask_to_feature_collection(
    mask: np.ndarray,
    meta: PreprocessMeta,
    cfg: PostprocessParams,
    *,
    source_name: str | None = None,
    class_name: str = "parking",
    probability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convert a binary model-space mask to cleaned EPSG:4326 GeoJSON."""
    unscaled = _resize_mask(mask, meta.width, meta.height)
    polys = list(_iter_polygons(unscaled, meta.transform))
    if not polys:
        return {"type": "FeatureCollection", "features": []}

    cleaned: list[BaseGeometry] = []
    for poly in polys:
        fixed = _drop_small_holes(poly, meta.crs, cfg.hole_area_m2).buffer(0)
        if fixed.is_empty:
            continue
        fixed = _simplify_geom(fixed, meta.crs, cfg.simplify_m)
        if area_m2(fixed, meta.crs) < cfg.min_area_m2:
            continue
        cleaned.append(fixed)

    if not cleaned:
        return {"type": "FeatureCollection", "features": []}

    merged = unary_union(cleaned)
    geom_list: list[BaseGeometry]
    if isinstance(merged, Polygon):
        geom_list = [merged]
    elif isinstance(merged, MultiPolygon):
        geom_list = list(merged.geoms)
    else:
        geom_list = [g for g in getattr(merged, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]

    features: list[dict[str, Any]] = []
    prob_unscaled: np.ndarray | None = None
    if probability is not None:
        prob_unscaled = np.asarray(
            Image.fromarray(np.asarray(probability, dtype=np.float32), mode="F").resize((meta.width, meta.height), Image.BILINEAR)
        )

    for geom in geom_list:
        props: dict[str, Any] = {
            "class_id": 1,
            "class_name": class_name,
            "source": source_name,
        }
        if prob_unscaled is not None:
            inside = rasterize(
                [(mapping(geom), 1)],
                out_shape=(meta.height, meta.width),
                transform=meta.transform,
                fill=0,
                dtype=np.uint8,
            )
            inside_values = prob_unscaled[inside == 1]
            if inside_values.size > 0:
                props["confidence"] = float(np.mean(inside_values))
        features.append(
            {
                "type": "Feature",
                "geometry": mapping_in_wgs84(geom, meta.crs),
                "properties": props,
            }
        )

    return {"type": "FeatureCollection", "features": features}
