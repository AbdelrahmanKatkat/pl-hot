import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin


@pytest.fixture()
def chips_dir(tmp_path: Path) -> Path:
    root = tmp_path / "chips"
    root.mkdir()
    transform = from_origin(500000, 1000, 1, 1)

    # x=0,8,16 → three (x//4) blocks so spatial_split can fill both train and val.
    for i, tile_x in enumerate((0, 8, 16)):
        arr = np.zeros((3, 32, 32), dtype=np.uint8)
        arr[0, 8:20, 8:20] = 200 + i
        arr[1, 8:20, 8:20] = 150
        arr[2, 8:20, 8:20] = 100
        p = root / f"OAM-{tile_x:04d}-0000-18.tif"
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            width=32,
            height=32,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            dst.write(arr)
    return root


@pytest.fixture()
def labels_dir(tmp_path: Path) -> Path:
    root = tmp_path / "labels"
    root.mkdir()
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x0, y0 = 500008.0, 992.0
    x1, y1 = 500020.0, 980.0
    ll = [to_wgs84.transform(x, y) for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0))]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ll],
                },
                "properties": {},
            }
        ],
    }
    (root / "labels.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    return root
