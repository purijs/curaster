"""
Shared fixtures for curaster chain-operation tests.

Creates small synthetic GeoTIFFs in EPSG:4326 centred over central Europe
so that reprojection to EPSG:32632 (UTM zone 32N) is well-conditioned.
"""

import json
import pytest
import numpy as np

try:
    from osgeo import gdal, osr
    _GDAL_OK = True
except ImportError:
    _GDAL_OK = False

try:
    import curaster
    _CURASTER_OK = True
except ImportError:
    _CURASTER_OK = False


# ── constants ────────────────────────────────────────────────────────────────

W, H   = 80, 80           # small but big enough for a 3×3 focal kernel
BANDS  = 4
# lon 10.0–10.8°, lat 50.0–50.8°  (central Europe – UTM zone 32N)
GT     = (10.0, 0.01, 0.0, 50.8, 0.0, -0.01)
EPSG_SRC = 4326
EPSG_DST = 32632


# ── module-level skip guards ─────────────────────────────────────────────────

if not _GDAL_OK:
    pytest.skip("osgeo.gdal not installed", allow_module_level=True)
if not _CURASTER_OK:
    pytest.skip("curaster not installed / not compiled", allow_module_level=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_tif(path: str, num_bands: int = BANDS,
              width: int = W, height: int = H) -> str:
    """Write a Float32 GeoTIFF; band b has all pixels = float(b) * 10."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, width, height, num_bands, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG_SRC)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform(GT)
    for b in range(1, num_bands + 1):
        data = np.full((height, width), float(b) * 10.0, dtype=np.float32)
        ds.GetRasterBand(b).WriteArray(data)
        ds.GetRasterBand(b).SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None
    return path


def _aoi_in_src_crs(margin: float = 0.2) -> str:
    """GeoJSON polygon covering the middle (1-2×margin) of the raster in EPSG:4326."""
    x0 = GT[0] + GT[1] * (W * margin)
    x1 = GT[0] + GT[1] * (W * (1 - margin))
    y0 = GT[3] + GT[5] * (H * margin)
    y1 = GT[3] + GT[5] * (H * (1 - margin))
    return json.dumps({
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
    })


def _aoi_in_dst_crs() -> str:
    """
    GeoJSON polygon in EPSG:32632 (UTM 32N) covering roughly the same
    geographic region as _aoi_in_src_crs().

    Approximate UTM 32N extents for lon=10.16-10.64, lat=50.16-50.64:
      easting  ~ 596000 – 628000 m
      northing ~ 5564000 – 5614000 m
    (computed once with pyproj; hardcoded here to avoid a runtime dependency)
    """
    x0, x1 = 596_000.0, 628_000.0
    y0, y1 = 5_564_000.0, 5_614_000.0
    return json.dumps({
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
    })


def read_tif_as_array(path: str) -> np.ndarray:
    """Return (bands, H, W) float32 ndarray from a GeoTIFF."""
    ds = gdal.Open(path)
    nb = ds.RasterCount
    h, w = ds.RasterYSize, ds.RasterXSize
    out = np.empty((nb, h, w), dtype=np.float32)
    for b in range(1, nb + 1):
        out[b - 1] = ds.GetRasterBand(b).ReadAsArray()
    ds = None
    return out


# ── session-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tif_4band(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "test_4band.tif")
    return _make_tif(path, num_bands=4)


@pytest.fixture(scope="session")
def tif_1band(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "test_1band.tif")
    return _make_tif(path, num_bands=1)


@pytest.fixture(scope="session")
def aoi_src():
    """AOI polygon in source CRS (EPSG:4326)."""
    return _aoi_in_src_crs()


@pytest.fixture(scope="session")
def aoi_dst():
    """AOI polygon in target CRS (EPSG:32632)."""
    return _aoi_in_dst_crs()
