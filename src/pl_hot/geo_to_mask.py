"""Rasterize GeoJSON parking polygons into chip-aligned masks."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import rasterize
from shapely.geometry import shape

from .georef import reproject_geometry


def _labels_crs_from_geojson(data: dict[str, Any]) -> str:
    """Best-effort CRS from a GeoJSON `crs` member. fAIr OSM labels are EPSG:4326."""
    crs_obj = data.get("crs")
    if not isinstance(crs_obj, dict):
        return "EPSG:4326"
    props = crs_obj.get("properties")
    if not isinstance(props, dict):
        return "EPSG:4326"
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "EPSG:4326"


def load_label_collection(labels_geojson: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Return (geometry dicts, source CRS). Load once; rasterize many chips."""
    data = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError("labels_geojson must be a FeatureCollection")
    geometries = [f["geometry"] for f in data.get("features", []) if f.get("geometry")]
    return geometries, _labels_crs_from_geojson(data)


def rasterize_labels_for_chip(
    labels_geojson: str | Path,
    chip_path: str | Path,
    *,
    src_crs: str | None = None,
    burn_value: int = 1,
    geometries: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Burn labels to a binary mask at the chip's **native** height×width (georef-aligned).

    Do not resize here. Training resizes the saved PNG with nearest-neighbor so
    class ids stay 0/1; bilinear is only for the RGB chip.
    """
    if geometries is None:
        geometries, detected_crs = load_label_collection(labels_geojson)
        src_crs = src_crs or detected_crs
    else:
        src_crs = src_crs or "EPSG:4326"

    with rasterio.open(chip_path) as src:
        chip_crs = src.crs
        transform = src.transform
        shape_hw = (src.height, src.width)

    if chip_crs is None:
        raise ValueError(f"Chip has no CRS: {chip_path}")

    projected = [
        reproject_geometry(shape(geom), CRS.from_user_input(src_crs), chip_crs)
        for geom in geometries
    ]
    burned = rasterize(
        [(g, burn_value) for g in projected if not g.is_empty],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return burned
