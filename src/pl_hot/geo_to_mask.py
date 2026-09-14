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


def _load_label_geometries(labels_geojson: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError("labels_geojson must be a FeatureCollection")
    return [f["geometry"] for f in data.get("features", []) if f.get("geometry")]


def rasterize_labels_for_chip(
    labels_geojson: str | Path,
    chip_path: str | Path,
    *,
    src_crs: str = "EPSG:4326",
    burn_value: int = 1,
) -> np.ndarray:
    """Burn labels to a binary mask aligned to one chip."""
    geometries = _load_label_geometries(labels_geojson)
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
