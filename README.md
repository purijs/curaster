# cuRaster
[![Build and Publish Release](https://github.com/purijs/curaster/actions/workflows/release.yml/badge.svg)](https://github.com/purijs/curaster/actions/workflows/release.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/curaster)](https://pypi.org/project/curaster/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curaster)](https://pypi.org/project/curaster/)

**cuRaster** is a high-performance Python library for GPU-accelerated raster processing. It reads GeoTIFF files (locally or directly from S3), executes band-math algebra on the GPU, and optionally reprojects, clips, and streams results — all through a clean, lazy pipeline API.

---

## Table of Contents

- [Installation](#installation)
- [Requirements (building from source)](#requirements-building-from-source)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [curaster.open()](#curasteropen)
  - [Chain.select_bands()](#chainselect_bands)
  - [Chain.persist()](#chainpersist)
  - [Chain.algebra()](#chainalgebra)
  - [Chain.clip()](#chainclip)
  - [Chain.reproject()](#chainreproject)
  - [Chain.get_info()](#chainget_info)
  - [Chain.save_local()](#chainsave_local)
  - [Chain.save_s3()](#chainsave_s3)
  - [Chain.to_memory()](#chainto_memory)
  - [Chain.iter_begin()](#chainiter_begin)
  - [Chain.focal()](#chainfocalstatmean-radius1-shapesquare-clamp_bordertrue)
  - [Chain.terrain()](#chainterrainmetrics-unitdegrees-sun_azimuth3150-sun_altitude450-methodhorn)
  - [Chain.texture()](#chaintexturefeatures-window11-levels32-direction_modeaverage-log_scalefalse-val_min00-val_max00)
  - [Chain.zonal_stats()](#chainzonal_statsgeojson-statsmean-std-min-max-count-sum-band1-verbosefalse)
  - [curaster.open_stack() / StackChain](#curasteropen_stackfiles--stackchain)
    - [StackChain.algebra()](#stackchainalgebra)
    - [StackChain.reproject()](#stackchainreproject)
  - [RasterResult](#rasterresult)
  - [ChunkQueue](#chunkqueue)
- [Examples](#examples)
- [Building from Source](#building-from-source)
- [License](#license)

---

## Installation

```bash
pip install curaster
```

Wheels are pre-built for Linux and Windows, Python 3.9–3.13. A compatible NVIDIA GPU and driver must be present at runtime (the CUDA runtime is bundled in the wheel).

---

## Requirements (building from source)

| Requirement | Notes |
|---|---|
| NVIDIA GPU | CUDA Compute Capability ≥ 7.5 (Turing+) |
| CUDA Toolkit 12.5+ | `nvcc` must be on `PATH` |
| GDAL 3.x | `libgdal-dev` on Linux, `gdal` conda package on Windows |
| CMake 3.18+ | |
| OpenSSL + libcurl | For direct S3 access |
| libzstd | For ZSTD-compressed GeoTIFF tiles |
| pybind11 | `pip install pybind11` |
| C++17 compiler | GCC 11+, MSVC 2022+ |

---

## Quick Start

```python
import curaster

# Compute NDVI and save to a local GeoTIFF
curaster.open("landsat.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .save_local("ndvi.tif")
```

---

## API Reference

### `curaster.open(path, bands=[])`

Open a GeoTIFF and return a lazy `Chain`. No GPU work happens here.

```python
chain = curaster.open("input.tif")
chain = curaster.open("/vsis3/my-bucket/data/scene.tif")  # S3 direct-read
chain = curaster.open("sentinel2.tif", bands=[2, 3, 4])   # select bands 2, 3, 4 (1-indexed)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Local file path or GDAL-like S3 URI (`/vsis3/`) |
| `bands` | `list[int]` | `[]` | 1-indexed band selection (empty = all bands) |

**Returns** `Chain`

---

### `Chain.select_bands(bands)`

Return a new Chain that processes only the specified source bands (1-indexed). Equivalent to passing `bands=` to `curaster.open()` but can also be called on derived chains.

```python
curaster.open("4band.tif").select_bands([1, 3]).clip(aoi).to_memory()
```

| Parameter | Type | Description |
|---|---|---|
| `bands` | `list[int]` | 1-indexed band numbers to keep |

**Returns** a new `Chain`

---

### `Chain.persist()`

Pre-load the source raster into GPU VRAM so that all subsequent operations on this chain skip disk or S3 I/O entirely. Useful when you run the same source raster through multiple independent pipelines (e.g. different algebra expressions or clip regions) and want to avoid re-reading it each time.

```python
# Load once into VRAM, then run multiple pipelines without re-reading
scene = curaster.open("/vsis3/bucket/sentinel2.tif").persist()

scene.algebra("(B8 - B4) / (B8 + B4)").save_local("ndvi.tif")
scene.algebra("(B3 - B8) / (B3 + B8)").save_local("ndwi.tif")
scene.algebra("(B11 - B8) / (B11 + B8)").clip(aoi).save_local("nbr.tif")
```

All chains derived from a persisted chain share the same VRAM data. Raises `RuntimeError` if the decoded raster exceeds 80% of available VRAM, or if raster dimensions exceed the CUDA 2D texture limit on the active GPU.

**Returns** a new `Chain`

---

### `Chain.algebra(expression)`

Append a band-math expression. Bands are referenced as `B1`, `B2`, … (1-indexed).

**`algebra` always produces a single-band output.** All bands referenced in the expression are combined into one scalar value per pixel. Any operation chained after `algebra` (clip, reproject, focal, etc.) receives a 1-band raster.

```python
chain.algebra("(B5 - B4) / (B5 + B4)")      # NDVI
chain.algebra("B1 * 0.0001")                  # Scale factor
chain.algebra("(B3 + B2 + B1) / 3")          # Visible mean
chain.algebra("B4 > 0.3")                     # Boolean mask (1.0 or 0.0)
chain.algebra("(B4 > 0.2) * B4")             # Apply mask conditionally
```

Supported operators: `+  -  *  /  >  <  >=  <=  ==  !=`

| Parameter | Type | Description |
|---|---|---|
| `expression` | `str` | Band-math expression string |

**Returns** a new `Chain` (original is unmodified)

---

### `Chain.clip(geojson)`

Clip the output to a polygon. Pixels outside the polygon are set to zero. The masking is applied on the GPU.

The geometry must be in the **CRS that is active at the point in the chain where `clip` appears**:

- `.clip(aoi).reproject(...)` — `aoi` must be in the **source CRS** (clip executes before reprojection)
- `.reproject(...).clip(aoi)` — `aoi` must be in the **target CRS** (clip executes after reprojection)

The two orderings are strictly different operations. Multiple `clip` calls accumulate — each geometry is intersected with the previous mask.

```python
import json

aoi = json.dumps({
    "type": "Polygon",
    "coordinates": [[[10.0, 52.0], [11.0, 52.0], [11.0, 53.0], [10.0, 53.0], [10.0, 52.0]]]
})

chain.algebra("(B5 - B4) / (B5 + B4)").clip(aoi)
```

| Parameter | Type | Description |
|---|---|---|
| `geojson` | `str` | GeoJSON string — `Polygon` or `MultiPolygon` in the raster's current CRS |

**Returns** a new `Chain`

---

### `Chain.reproject(target_crs, ...)`

Reproject the output to a different coordinate reference system.

```python
chain.reproject("EPSG:4326")                             # Auto pixel size
chain.reproject("EPSG:3857", res_x=10.0, res_y=10.0)   # Fixed 10 m resolution
chain.reproject("EPSG:4326", resampling="nearest")      # Nearest-neighbour

# Fixed output extent (in target CRS units)
chain.reproject(
    "EPSG:4326",
    res_x=0.0001, res_y=0.0001,
    te_xmin=9.5, te_ymin=51.5,
    te_xmax=10.5, te_ymax=52.5
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_crs` | `str` | required | Any CRS string GDAL understands (EPSG code, WKT, PROJ string) |
| `res_x` | `float` | `0` | Output pixel width in target CRS units (0 = auto-derive) |
| `res_y` | `float` | `0` | Output pixel height in target CRS units (0 = auto-derive) |
| `resampling` | `str` | `"bilinear"` | `"bilinear"` or `"nearest"` |
| `nodata` | `float` | `-9999.0` | Fill value for pixels outside the source extent |
| `te_xmin` | `float` | `0` | Output extent — min X in target CRS |
| `te_ymin` | `float` | `0` | Output extent — min Y in target CRS |
| `te_xmax` | `float` | `0` | Output extent — max X in target CRS |
| `te_ymax` | `float` | `0` | Output extent — max Y in target CRS |

**Returns** a new `Chain`

---

### `Chain.get_info()`

Return metadata for the output raster without executing the pipeline.

```python
info = curaster.open("scene.tif").reproject("EPSG:4326").get_info()
print(info)
# {'width': 4096, 'height': 3072, 'geotransform': [...], 'crs': 'GEOGCS[...]'}
```

**Returns** `dict` with keys `width`, `height`, `geotransform` (list of 6 floats), `crs` (WKT string)

---

### `Chain.save_local(path, verbose=False)`

Execute the pipeline and write a Float32 tiled GeoTIFF to disk.

```python
curaster.open("scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .save_local("ndvi.tif", verbose=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Output file path |
| `verbose` | `bool` | `False` | Print a GDAL-style progress bar |

---

### `Chain.save_s3(s3_path, verbose=False)`

Execute the pipeline and upload the result directly to S3.  
AWS credentials must be set via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`).

```python
curaster.open("scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .save_s3("/vsis3/my-bucket/output/ndvi.tif")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `s3_path` | `str` | required | Upload destination (`/vsis3/bucket/key`) |
| `verbose` | `bool` | `False` | Print a progress bar |

---

### `Chain.to_memory(verbose=False)`

Execute and return all pixels as a `RasterResult` object. Raises `RuntimeError` if the result would exceed a dynamically-computed RAM budget (8–70 % of available RAM depending on the operation type — focal/terrain/texture reserve more headroom for their GPU buffers). Use `iter_begin()` for large rasters or when memory is tight.

```python
result = curaster.open("scene.tif") \
    .algebra("B1 * 0.0001") \
    .to_memory()

import numpy as np
arr = result.data()          # shape (height, width) for 1-band, (bands, height, width) for multi-band
print(arr.shape, arr.dtype)  # float32
print(result.width, result.height, result.bands, result.proj)
```

**Returns** `RasterResult`

---

### `Chain.iter_begin(buf_chunks=4)`

Start background execution and return a `ChunkQueue` for memory-efficient streaming. Each chunk covers a horizontal strip of the output.

```python
queue = curaster.open("huge_scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .iter_begin(buf_chunks=8)

while True:
    chunk = queue.next()
    if chunk is None:
        break
    # chunk["data"] shape: (height, width) for 1-band  |  (bands, height, width) for multi-band
    process(chunk["data"], chunk["y_offset"])
```

All output operations (`to_memory`, `save_local`, `save_s3`, `iter_begin`) respect the output band count of the pipeline:
- No compute op: passthrough — all source bands (or those selected via `select_bands` / `bands=`) streamed as-is
- `algebra`: always 1 band
- `focal`: one band per input band (independent per-band window)
- `terrain`: one band per requested metric
- `texture`: 18 bands (average mode) or 72 bands (separate mode)

Clip and reproject are applied per-band and do not change the band count.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `buf_chunks` | `int` | `4` | Number of completed chunks to buffer before backpressure |

**Returns** `ChunkQueue`

---

### `RasterResult`

Returned by `to_memory()`.

| Attribute / Method | Type | Description |
|---|---|---|
| `.width` | `int` | Output width in pixels |
| `.height` | `int` | Output height in pixels |
| `.bands` | `int` | Number of output bands |
| `.proj` | `str` | WKT coordinate reference system |
| `.data()` | `np.ndarray` | `float32` array — shape `(height, width)` for 1 band, `(bands, height, width)` for multi-band |

---

### `ChunkQueue`

Returned by `iter_begin()`. Processing runs on a background thread.

| Method | Returns | Description |
|---|---|---|
| `.next()` | `dict` or `None` | Pop the next chunk, or `None` on completion |

Each chunk dict has keys: `y_offset` (`int`), `width` (`int`), `height` (`int`), `data` (`np.ndarray float32` — shape `(height, width)` or `(bands, height, width)`).

---

## Examples

### NDVI — local file, save to disk

```python
import curaster

curaster.open("landsat8_sr.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .save_local("ndvi.tif", verbose=True)
```

---

### S3 direct-read → S3 write

```python
import curaster, os

# Credentials are read from the environment automatically
curaster.open("/vsis3//my-bucket/scenes/LC08_2024_scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .save_s3("/vsis3/my-bucket/output/ndvi.tif")
```

---

### Clip to area of interest

```python
import curaster, json

aoi = json.dumps({
    "type": "Polygon",
    "coordinates": [[[13.3, 52.4], [13.5, 52.4], [13.5, 52.6], [13.3, 52.6], [13.3, 52.4]]]
})

curaster.open("sentinel2.tif") \
    .algebra("(B8 - B4) / (B8 + B4)") \
    .clip(aoi) \
    .save_local("ndvi_berlin.tif")
```

---

### Reproject to WGS84 with fixed resolution

```python
import curaster

curaster.open("utm_scene.tif") \
    .algebra("(B4 - B3) / (B4 + B3)") \
    .reproject("EPSG:4326", res_x=0.0001, res_y=0.0001) \
    .save_local("ndvi_wgs84.tif")
```

---

### Multiband passthrough — clip and reproject without algebra

Chains without `algebra`, `terrain`, or `texture` pass all source bands through untouched.

```python
import curaster, json

aoi = json.dumps({
    "type": "Polygon",
    "coordinates": [[[13.3, 52.4], [13.5, 52.4], [13.5, 52.6], [13.3, 52.6], [13.3, 52.4]]]
})

# Clip a 4-band raster: returns (4, H, W)
result = curaster.open("sentinel2_4band.tif").clip(aoi).to_memory()
print(result.data().shape)   # (4, H, W)

# Reproject all bands to WGS84
curaster.open("utm_scene.tif") \
    .reproject("EPSG:4326", res_x=0.0001, res_y=0.0001) \
    .save_local("reprojected_all_bands.tif")

# Select specific bands (1-indexed)
curaster.open("sentinel2.tif", bands=[2, 3, 4]) \
    .clip(aoi) \
    .save_local("rgb_clipped.tif")

# Equivalent using select_bands
curaster.open("sentinel2.tif") \
    .select_bands([2, 3, 4]) \
    .reproject("EPSG:4326") \
    .iter_begin()
```

---

### Full pipeline: S3 → algebra → clip → reproject → S3

```python
import curaster, json

aoi = json.dumps({
    "type": "Polygon",
    "coordinates": [[[10.0, 52.0], [11.0, 52.0], [11.0, 53.0], [10.0, 53.0], [10.0, 52.0]]]
})

curaster.open("/vsis3/my-bucket/raw/sentinel2.tif") \
    .algebra("(B8 - B4) / (B8 + B4)") \
    .clip(aoi) \
    .reproject("EPSG:4326", res_x=0.0001, res_y=0.0001) \
    .save_s3("/vsis3/my-bucket/processed/ndvi_reprojected.tif")
```

---

### Inspect output metadata before running

```python
import curaster

info = curaster.open("scene.tif") \
    .reproject("EPSG:4326", res_x=0.0001) \
    .get_info()

print(f"Output will be {info['width']} × {info['height']} pixels")
print(f"CRS: {info['crs'][:60]}...")
```

---

### Load into numpy / xarray

```python
import curaster
import numpy as np
import xarray as xr

result = curaster.open("scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .to_memory()

arr = result.data()   # shape (H, W) for 1-band; (bands, H, W) for multi-band; dtype float32
arr[arr == -9999.0] = np.nan

gt = curaster.open("scene.tif").get_info()["geotransform"]
xcoords = gt[0] + np.arange(result.width)  * gt[1]
ycoords = gt[3] + np.arange(result.height) * gt[5]

da = xr.DataArray(arr, dims=["y", "x"], coords={"x": xcoords, "y": ycoords})
print(da)
```

---

### Streaming large rasters chunk-by-chunk

```python
import curaster
import numpy as np

output = np.zeros((10000, 10000), dtype=np.float32)

queue = curaster.open("massive_scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .iter_begin(buf_chunks=6)

while True:
    chunk = queue.next()
    if chunk is None:
        break
    y0 = chunk["y_offset"]
    h  = chunk["height"]
    output[y0 : y0 + h, :] = chunk["data"]

print("Done. Mean NDVI:", output.mean())
```

---

### Boolean / conditional expression

```python
import curaster

# Mask pixels where NIR reflectance > 0.3, zero elsewhere
curaster.open("scene.tif") \
    .algebra("(B5 > 0.3) * B5") \
    .save_local("nir_high_mask.tif")

# Multi-band composite score
curaster.open("scene.tif") \
    .algebra("(B5 - B4) / (B5 + B4) + (B3 - B2) / (B3 + B2)") \
    .save_local("composite_score.tif")
```

---

### Focal — multi-band independent smoothing

```python
import curaster

# 4-band input → 4-band smoothed output (one focal pass per band)
curaster.open("multispectral.tif") \
    .focal("mean", radius=3) \
    .save_local("smoothed_4band.tif")

# Focal → clip (clip is applied in the source CRS; all bands preserved)
curaster.open("multispectral.tif") \
    .focal("mean", radius=3) \
    .clip('{"type":"Polygon","coordinates":[[[10.2,50.2],[10.6,50.2],[10.6,50.6],[10.2,50.6],[10.2,50.2]]]}') \
    .save_local("smoothed_clipped.tif")

# Focal → reproject (reproject is applied after all band focal passes)
curaster.open("multispectral.tif") \
    .focal("mean", radius=3) \
    .reproject("EPSG:32632", res_x=1000, res_y=1000) \
    .save_local("smoothed_utm.tif")

# Focal → clip (source CRS) → reproject to UTM
curaster.open("multispectral.tif") \
    .focal("median", radius=2, shape="circle") \
    .clip('{"type":"Polygon","coordinates":[[[10.2,50.2],[10.6,50.2],[10.6,50.6],[10.2,50.6],[10.2,50.2]]]}') \
    .reproject("EPSG:32632") \
    .save_local("smoothed_clipped_utm.tif")

# Stream multi-band focal result chunk by chunk
chain = curaster.open("multispectral.tif").focal("mean", radius=5)
queue = chain.iter_begin(buf_chunks=4)
while True:
    chunk = queue.next()
    if chunk is None:
        break
    # chunk["data"] shape: (bands, height, width)
    print(f"offset={chunk['y_offset']} bands={chunk['data'].shape[0]}")
```

---

### Clip ordering — source CRS vs target CRS

Clip geometry must be in the **same CRS as the data at that point in the chain**.

```python
import curaster

src_aoi = '{"type":"Polygon","coordinates":[[[10.2,50.2],[10.6,50.2],[10.6,50.6],[10.2,50.6],[10.2,50.2]]]}'
utm_aoi = '{"type":"Polygon","coordinates":[[[596000,5564000],[628000,5564000],[628000,5614000],[596000,5614000],[596000,5564000]]]}'

# .clip().reproject() — geometry in SOURCE CRS (EPSG:4326)
# Clips first, then reprojects the clipped raster to UTM
curaster.open("scene.tif") \
    .clip(src_aoi) \
    .reproject("EPSG:32632") \
    .save_local("clip_then_reproject.tif")

# .reproject().clip() — geometry in TARGET CRS (EPSG:32632 UTM)
# Reprojects the full raster first, then clips the UTM output
curaster.open("scene.tif") \
    .reproject("EPSG:32632") \
    .clip(utm_aoi) \
    .save_local("reproject_then_clip.tif")
```

---

### Zonal statistics

```python
import json
import curaster

zones_fc = json.dumps({
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "zone_a",
         "geometry": {"type": "Polygon", "coordinates": [[[10.2,50.2],[10.5,50.2],[10.5,50.5],[10.2,50.5],[10.2,50.2]]]}},
        {"type": "Feature", "id": "zone_b",
         "geometry": {"type": "Polygon", "coordinates": [[[10.5,50.5],[10.8,50.5],[10.8,50.8],[10.5,50.8],[10.5,50.5]]]}}
    ]
})

# Whole-image stats (single band)
results = curaster.open("dem.tif").zonal_stats()
for r in results:
    print(r.to_dict())
# → [{"zone_id": "all", "mean": ..., "min": ..., "max": ..., "std_dev": ..., "count": ..., "sum": ...}]

# Per-zone stats on band 1
results = curaster.open("multispectral.tif").zonal_stats(
    stats=["mean", "std", "count"],
    band=1,
    geojson_str=zones_fc
)
for r in results:
    d = r.to_dict()
    print(f"zone={d['zone_id']}  mean={d['mean']:.4f}  std={d['std_dev']:.4f}  n={d['count']}")

# Zonal stats on algebra output (NDVI), band=1 (the only output band)
results = curaster.open("multispectral.tif") \
    .algebra("(B5 - B4) / (B5 + B4)") \
    .zonal_stats(stats=["mean", "min", "max"], band=1, geojson_str=zones_fc)

# Serialise to list of dicts (e.g. for pandas / JSON export)
rows = [r.to_dict() for r in results]
import pandas as pd
df = pd.DataFrame(rows)
```

| `ZoneResult` attribute | Type | Description |
|---|---|---|
| `zone_id` | `str` | Feature `id` from GeoJSON, or `"all"` for whole-image |
| `mean` | `float` | Band mean over valid (non-nodata) pixels |
| `min` | `float` | Minimum pixel value |
| `max` | `float` | Maximum pixel value |
| `std_dev` | `float` | Standard deviation |
| `count` | `int` | Number of valid pixels |
| `sum` | `float` | Sum of valid pixels |
| `to_dict()` | `dict` | All of the above as a Python dict |

---

### `StackChain` — multi-scene algebra and reprojection

```python
import curaster

# Stack three Sentinel-2 scenes (co-registered, same grid)
stack = curaster.open_stack([
    "S2_20240601.tif",
    "S2_20240615.tif",
    "S2_20240701.tif",
])

# Per-pixel temporal mean across scenes, then reproject to UTM
stack.temporal("mean") \
     .reproject("EPSG:32632", res_x=10, res_y=10) \
     .save_local("seasonal_mean_utm.tif")

# Algebra on the stack (B1 = first scene band 1, etc. — use scene index prefix)
# then save locally
stack.algebra("(B5 - B4) / (B5 + B4)") \
     .save_local("ndvi_stack.tif")

# Reproject all scenes in the stack before temporal reduction
reproj_stack = stack.reproject("EPSG:32632", res_x=10, res_y=10)
result = reproj_stack.temporal("std")
result.save_local("temporal_std_utm.tif")
```

---

---

### `Chain.focal(stat="mean", radius=1, shape="square", clamp_border=True)`

Apply a moving-window focal statistic. On a **multi-band input, focal runs independently on each band** and returns the same number of bands as the input. You can follow `focal` with `clip`, `reproject`, `save_local`, etc. and all bands are preserved.

```python
# Single-band DEM smoothing
curaster.open("dem.tif") \
    .focal("mean", radius=5) \
    .save_local("dem_smoothed.tif")

curaster.open("dem.tif") \
    .focal("median", radius=3, shape="circle") \
    .save_local("dem_median.tif")

# Multi-band: 4-band input → 4-band focal output
curaster.open("sentinel2_4band.tif") \
    .focal("mean", radius=3) \
    .save_local("smoothed_4band.tif")

# Focal + clip (post-focal clip preserves all bands)
import json
aoi = json.dumps({"type": "Polygon", "coordinates": [[[10.0, 52.0], [11.0, 52.0], [11.0, 53.0], [10.0, 53.0], [10.0, 52.0]]]})
curaster.open("sentinel2_4band.tif") \
    .focal("mean") \
    .clip(aoi) \
    .save_local("smoothed_clipped_4band.tif")

# Focal + reproject (post-focal reproject preserves all bands)
curaster.open("sentinel2_4band.tif") \
    .focal("mean", radius=3) \
    .reproject("EPSG:4326") \
    .save_local("smoothed_reprojected_4band.tif")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stat` | `str` | `"mean"` | `mean`, `sum`, `min`, `max`, `std`, `variance`, `median`, `range` |
| `radius` | `int` | `1` | Half-window radius in pixels (window = 2R+1 × 2R+1) |
| `shape` | `str` | `"square"` | `"square"` or `"circle"` |
| `clamp_border` | `bool` | `True` | Clamp border pixels (replicate edge rows/cols) |

**Returns** a new `Chain`

---

### `Chain.terrain(metrics=[], unit="degrees", sun_azimuth=315.0, sun_altitude=45.0, method="horn")`

Compute terrain derivatives from a DEM.

```python
curaster.open("dem.tif") \
    .terrain(["slope", "aspect", "hillshade"]) \
    .save_local("terrain.tif")

# All supported metrics:
curaster.open("dem.tif") \
    .terrain(["slope", "aspect", "hillshade", "tri", "tpi",
              "roughness", "prof_curv", "plan_curv", "total_curv"],
             unit="degrees") \
    .save_local("terrain_all.tif")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metrics` | `list[str]` | `[]` = all | Any subset of: `slope`, `aspect`, `hillshade`, `tri`, `tpi`, `roughness`, `prof_curv`, `plan_curv`, `total_curv` (empty = all metrics) |
| `unit` | `str` | `"degrees"` | Slope output unit — `"degrees"`, `"radians"`, or `"percent"` |
| `sun_azimuth` | `float` | `315.0` | Sun azimuth for hillshade (degrees from north) |
| `sun_altitude` | `float` | `45.0` | Sun altitude for hillshade (degrees above horizon) |
| `method` | `str` | `"horn"` | Gradient method: `"horn"` (3×3 weighted) or `"zevenbergen"` (2-point central) |

Output is a multi-band GeoTIFF with one band per metric in the order given.

**Returns** a new `Chain`

---

### `Chain.texture(features=[], window=11, levels=32, direction_mode="average", log_scale=False, val_min=0.0, val_max=0.0)`

Compute GLCM (Grey-Level Co-occurrence Matrix) Haralick texture features.

```python
curaster.open("sar.tif") \
    .texture(["contrast", "homogeneity", "entropy"], window=15, levels=64, log_scale=True) \
    .save_local("texture.tif")

# All 18 features, 4-direction average
curaster.open("image.tif") \
    .texture(window=11, levels=32) \
    .save_local("texture_full.tif")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `features` | `list[str]` | `[]` = all | Subset of the 18 Haralick features: `asm`, `contrast`, `correlation`, `variance`, `homogeneity`, `sum_average`, `sum_variance`, `sum_entropy`, `entropy`, `diff_variance`, `diff_entropy`, `dissimilarity`, `autocorrelation`, `max_probability`, `cluster_shade`, `cluster_prominence`, `imc1`, `imc2` |
| `window` | `int` | `11` | Sliding window size in pixels (forced odd) |
| `levels` | `int` | `32` | Grey level quantization levels |
| `direction_mode` | `str` | `"average"` | `"average"` — average 4 directions (18 output bands); `"separate"` — 4×18 = 72 output bands |
| `log_scale` | `bool` | `False` | Apply `10·log10(v)` before quantization (for SAR data) |
| `val_min` | `float` | `0.0` | Min input value for quantization (0,0 = auto-detect from file) |
| `val_max` | `float` | `0.0` | Max input value for quantization (0,0 = auto-detect from file) |

**Returns** a new `Chain`

---

### `Chain.zonal_stats(geojson_str="", stats=["mean", "std", "min", "max", "count", "sum"], band=1, verbose=False)`

Compute per-polygon zonal statistics over any raster. Terminal operation — returns results immediately. All preceding pipeline operations (algebra, clip, terrain, etc.) are applied before statistics are computed.

```python
import json, curaster

zones = json.dumps({
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[13.3, 52.4], [13.5, 52.4], [13.5, 52.6], [13.3, 52.6], [13.3, 52.4]]]}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[14.0, 52.0], [14.5, 52.0], [14.5, 52.5], [14.0, 52.5], [14.0, 52.0]]]}}
    ]
})

# Zonal stats on raw raster
results = curaster.open("ndvi.tif").zonal_stats(geojson_str=zones, stats=["mean", "std", "min", "max"])
for r in results:
    print(r.zone_id, r.mean, r.std_dev, r.min, r.max)

# Attribute access
print(r.count, r.sum)

# Convert to dict
d = r.to_dict()
# {'zone_id': 1, 'mean': 0.42, 'min': 0.01, 'max': 0.89, 'std_dev': 0.12, 'count': 4096, 'sum': 1720.3}

# Zonal stats after algebra (operates on the computed output)
results = (curaster.open("sentinel2.tif")
    .algebra("(B8 - B4) / (B8 + B4)")
    .zonal_stats(geojson_str=zones, stats=["mean", "min", "max"]))

# Whole-image stats (no geojson)
results = curaster.open("dem.tif").zonal_stats(stats=["mean", "min", "max"])
print(results[0].mean)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `geojson_str` | `str` | `""` | GeoJSON `FeatureCollection`, `Polygon`, or `MultiPolygon` string (empty = whole-image stats as a single zone) |
| `stats` | `list[str]` | all | Any subset of `mean`, `std`, `min`, `max`, `count`, `sum` |
| `band` | `int` | `1` | 1-indexed band to compute statistics for |
| `verbose` | `bool` | `False` | Print progress |

**Returns** `list[ZoneResult]`

#### `ZoneResult`

| Attribute / Method | Type | Description |
|---|---|---|
| `.zone_id` | `int` | 1-based zone index (order of features in the GeoJSON) |
| `.mean` | `float` | Mean pixel value |
| `.min` | `float` | Minimum pixel value |
| `.max` | `float` | Maximum pixel value |
| `.std_dev` | `float` | Standard deviation |
| `.count` | `int` | Number of valid pixels in the zone |
| `.sum` | `float` | Sum of pixel values |
| `.to_dict()` | `dict` | All of the above as a Python dict |

---

### `curaster.open_stack(files)` / `StackChain`

Open a multi-temporal stack of aligned GeoTIFF files and reduce them to a single raster.

```python
import curaster

stack = curaster.open_stack(["s2_20230601.tif", "s2_20230701.tif", "s2_20230801.tif"])

# Temporal difference (last - first)
stack.temporal("diff").save_local("diff.tif")

# Linear trend slope (change per scene)
stack.temporal("trend", time_values=[0.0, 30.0, 60.0]).save_local("trend.tif")

# Mean of all scenes
stack.temporal("mean").save_local("mean.tif")

# You can chain further operations after temporal reduction
stack.temporal("diff") \
    .clip(aoi_geojson) \
    .save_local("diff_clipped.tif")
```

| `temporal()` parameter | Type | Default | Description |
|---|---|---|---|
| `op` | `str` | required | `diff`, `ratio`, `anomaly_mean`, `trend`, `mean`, `std`, `min`, `max` |
| `t0` | `int` | `0` | Index of the first scene (for `diff`, `ratio`) |
| `t1` | `int` | `-1` | Index of the second scene (-1 = last) |
| `baseline` | `str` | `"mean"` | Baseline method for anomaly operations |
| `time_values` | `list[float]` | `[]` | Timestamps for `trend` (defaults to 0, 1, 2, …) |

**`StackChain.temporal()` Returns** a `Chain` (can chain `algebra`, `clip`, `reproject`, `save_local`, etc.)

All scenes in the stack must have the same width, height, and CRS. Use `.reproject()` on each `Chain` before stacking if misaligned.

---

#### `StackChain.algebra()`

Apply a band-math expression across all bands of the stack simultaneously. Bands are indexed by their physical position across all stacked scenes.

```python
stack = curaster.open_stack(["b1.tif", "b2.tif"])
# B1 = band 1 of scene 0, B2 = band 1 of scene 1, etc.
stack.algebra("(B2 - B1) / (B2 + B1)").save_local("ratio.tif")
```

Same syntax as `Chain.algebra()` — see [Chain.algebra()](#chainalgebra).

---

#### `StackChain.reproject()`

Reproject the entire stack before reduction. Accepts the same parameters as `Chain.reproject()` except `te_*` extent clipping is not supported.

```python
stack = curaster.open_stack(["utm32_a.tif", "utm32_b.tif"])
stack.reproject("EPSG:4326", res_x=0.0001, res_y=0.0001) \
     .temporal("mean") \
     .save_local("mean_wgs84.tif")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_crs` | `str` | required | Target CRS (EPSG code, WKT, PROJ string) |
| `res_x` | `float` | `0` | Output pixel width in target CRS units |
| `res_y` | `float` | `0` | Output pixel height in target CRS units |
| `resampling` | `str` | `"bilinear"` | `"bilinear"` or `"nearest"` |
| `nodata` | `float` | `-9999.0` | Fill value for pixels outside the source extent |

---

## Performance & Benchmarks

The following benchmarks demonstrate cuRaster's performance for both **Local** storage and direct **S3** reads.

**Hardware Specifications:**
- **Instance:** AWS g4dn.xlarge
- **Compute:** 4 vCPUs, 16 GiB RAM (Intel Xeon 2.5 GHz)
- **GPU:** 1x NVIDIA T4 Tensor Core (16 GiB VRAM, Compute Capability 7.5)
- **On-Demand Cost:** €0.563 / hour

We test across varying raster sizes:
- **S:** 2048 × 2048
- **M:** 4096 × 4096
- **L:** 8192 × 8192
- **XL:** 16384 × 16384
- **XXL:** 32768 × 8192

Each cell displays the **Processing Time** alongside the **Estimated Compute Cost** for that single operation.

### Local GeoTIFF Operations
These tests read files directly from the local NVMe SSD.

| Operation | S (2048×2048) | M (4096×4096) | L (8192×8192) | XXL (32768×8192) | XL (16384×16384) |
|---|---|---|---|---|---|
| A. Band Algebra (NDVI) | 79.9 ms | 309.7 ms | 1.14 s | 4.35 s | 4.56 s |
| B. Polygon Clip | 49.2 ms | 193.8 ms | 740.0 ms | 2.94 s | 2.91 s |
| C. Reprojection | 76.4 ms | 332.6 ms | 1.34 s | 7.19 s | 6.22 s |
| D. Full Pipeline (A+B+C) | 153.1 ms | 613.9 ms | 2.45 s | 10.94 s | 9.38 s |
| E. Large-file Stream | — | 231.3 ms | 957.0 ms | 3.64 s | 4.00 s |
| F. Multi-band Composite | 147.9 ms | 602.0 ms | 2.38 s | 8.54 s | 8.73 s |
| G. Boolean Spectral Mask | 75.2 ms | 292.7 ms | 1.18 s | 4.44 s | 4.58 s |
| H. Focal Median | 302.6 ms | 1.22 s | 7.56 s | 18.34 s | 18.59 s |
| I. Terrain (Slope+Aspect) | 74.4 ms | 356.8 ms | 4.05 s | 6.11 s | 6.02 s |
| K. Zonal Stats | 59.5 ms | 244.2 ms | 1.03 s | 5.95 s | 4.01 s |
| L. Temporal Stack | 495.5 ms | 2.09 s | 8.02 s | 23.63 s | 24.27 s |

### S3 Direct-Read Operations
These tests read data dynamically over the network from an AWS S3 bucket using GDAL's virtual file system and libcurl with HTTP Range requests.

| Operation | M (4096×4096) | L (8192×8192) | XL (16384×16384) |
|---|---|---|---|
| S3-A. Algebra | 249.1 ms | 984.6 ms | 2.10 s |
| S3-B. Reprojection | 114.3 ms | 437.4 ms | 2.55 s |
| S3-C. Full Pipeline | 146.6 ms | 2.63 s | 3.09 s |
| S3-D. Streaming | 235.9 ms | 886.4 ms | 1.08 s |
| S3-E. Focal | 4.25 s | 15.48 s | 17.29 s [iter] |
| S3-F. Terrain | 3.27 s | 11.06 s | 8.23 s |
| S3-H. Zonal | 3.20 s | 11.95 s | 10.60 s |
| S3-I. Temporal | 4.35 s | 15.57 s | 73.92 s |

---

## Building from Source

```bash
# 1. Install Python build dependencies
pip install build pybind11 scikit-build-core setuptools_scm

# 2. Install system libraries (Ubuntu/Debian)
sudo apt install libgdal-dev libzstd-dev libssl-dev libcurl4-openssl-dev libomp-dev

# 3. Build the wheel
python -m build --wheel

# 4. Install the built wheel
pip install wheelhouse/*.whl
```

Or build directly with CMake for development:

```bash
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)

# Copy the .so into your working directory
cp curaster*.so ..
```

---

## License

See [LICENSE](LICENSE) for details.
