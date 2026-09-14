"""Helpers for CRS transforms and GeoJSON conversions."""

from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


def reproject_geometry(geom: BaseGeometry, src_crs: Any, dst_crs: Any = "EPSG:4326") -> BaseGeometry:
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    if src == dst:
        return geom
    fn = Transformer.from_crs(src, dst, always_xy=True).transform
    return transform(fn, geom)


def area_m2(geom: BaseGeometry, crs: Any) -> float:
    """Area in square meters irrespective of source CRS axis units."""
    src = CRS.from_user_input(crs)
    if src.is_projected and src.axis_info and src.axis_info[0].unit_name.lower() in {"metre", "meter"}:
        return float(geom.area)
    fn = Transformer.from_crs(src, "EPSG:3857", always_xy=True).transform
    return float(transform(fn, geom).area)


def mapping_in_wgs84(geom: BaseGeometry, src_crs: Any) -> dict[str, Any]:
    return mapping(reproject_geometry(geom, src_crs, "EPSG:4326"))


def shape_from_geojson(geojson_geom: dict[str, Any], src_crs: Any = "EPSG:4326", dst_crs: Any = None) -> BaseGeometry:
    geom = shape(geojson_geom)
    if dst_crs is None:
        return geom
    return reproject_geometry(geom, src_crs, dst_crs)
