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
  - [Chain.algebra()](#chainalgebra)
  - [Chain.clip()](#chainclip)
  - [Chain.reproject()](#chainreproject)
  - [Chain.get_info()](#chainget_info)
  - [Chain.save_local()](#chainsave_local)
  - [Chain.save_s3()](#chainsave_s3)
  - [Chain.to_memory()](#chainto_memory)
  - [Chain.iter_begin()](#chainiter_begin)
  - [Chain.focal()](#chainfocalstat-radius3-shapesquare-clamp_bordertrue)
  - [Chain.terrain()](#chainterrainmetricsslope-unitdegrees-sun_azimuth3150-sun_altitude450-methodhorn)
  - [Chain.texture()](#chaintexturefeatures-window11-levels32-direction_modeaverage-log_scalefalse-val_min00-val_max00)
  - [Chain.zonal_stats()](#chainzonal_statsgeojson-statsmean-std-min-max-count-sum-band1-verbosefalse)
  - [curaster.open_stack() / StackChain](#curasteropen_stackfiles--stackchain)
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

### `curaster.open(path)`

Open a GeoTIFF and return a lazy `Chain`. No GPU work happens here.

```python
chain = curaster.open("input.tif")
chain = curaster.open("/vsis3/my-bucket/data/scene.tif") # S3 direct-read using GDAL vsis3 URI
```

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Local file path or GDAL-like S3 URI (`/vsis3/`) |

**Returns** `Chain`

---

### `Chain.algebra(expression)`

Append a band-math operation. Bands are referenced as `B1`, `B2`, … (1-indexed).

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

Clip the output to a polygon. Pixels outside the polygon are set to zero.

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
| `geojson` | `str` | GeoJSON string — `Polygon` or `MultiPolygon` |

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

Execute and return all pixels as a `RasterResult` object. Raises `RuntimeError` if the result would exceed 75 % of available RAM — use `iter_begin()` for large rasters.

```python
result = curaster.open("scene.tif") \
    .algebra("B1 * 0.0001") \
    .to_memory()

import numpy as np
arr = result.data()          # numpy array, shape (height, width), dtype float32
print(arr.mean(), arr.std())
print(result.width, result.height, result.proj)
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
    # chunk = {'y_offset': int, 'width': int, 'height': int, 'data': np.ndarray}
    process(chunk["data"], chunk["y_offset"])
```

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
| `.proj` | `str` | WKT coordinate reference system |
| `.data()` | `np.ndarray` | `float32` array of shape `(height, width)` |

---

### `ChunkQueue`

Returned by `iter_begin()`. Processing runs on a background thread.

| Method | Returns | Description |
|---|---|---|
| `.next()` | `dict` or `None` | Pop the next chunk, or `None` on completion |

Each chunk dict has keys: `y_offset` (`int`), `width` (`int`), `height` (`int`), `data` (`np.ndarray float32`).

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

arr = result.data()   # shape (H, W), dtype float32
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

---

### `Chain.focal(stat, radius=3, shape="square", clamp_border=True)`

Apply a moving-window focal statistic.

```python
curaster.open("dem.tif") \
    .focal("mean", radius=5) \
    .save_local("dem_smoothed.tif")

curaster.open("dem.tif") \
    .focal("median", radius=3, shape="circle") \
    .save_local("dem_median.tif")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stat` | `str` | required | `mean`, `sum`, `min`, `max`, `std`, `variance`, `median`, `range` |
| `radius` | `int` | `3` | Half-window radius in pixels (window = 2R+1 × 2R+1) |
| `shape` | `str` | `"square"` | `"square"` or `"circle"` |
| `clamp_border` | `bool` | `True` | Clamp border pixels (replicate edge rows/cols) |

**Returns** a new `Chain`

---

### `Chain.terrain(metrics=["slope"], unit="degrees", sun_azimuth=315.0, sun_altitude=45.0, method="horn")`

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
| `metrics` | `list[str]` | `["slope"]` | Any subset of: `slope`, `aspect`, `hillshade`, `tri`, `tpi`, `roughness`, `prof_curv`, `plan_curv`, `total_curv` |
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

### `Chain.zonal_stats(geojson, stats=["mean", "std", "min", "max", "count", "sum"], band=1, verbose=False)`

Compute per-polygon zonal statistics over any raster. Terminal operation — returns results immediately.

```python
import json, curaster

aoi = json.dumps({
    "type": "MultiPolygon",
    "coordinates": [...]
})

results = curaster.open("ndvi.tif").zonal_stats(aoi, stats=["mean", "std", "min", "max"])
for r in results:
    print(r.zone_id, r.mean, r.std_dev, r.min, r.max)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `geojson` | `str` | required | GeoJSON `Polygon` or `MultiPolygon` string |
| `stats` | `list[str]` | all | Any subset of `mean`, `std`, `min`, `max`, `count`, `sum` |
| `band` | `int` | `1` | Band number to compute statistics for (1-indexed) |
| `verbose` | `bool` | `False` | Print progress |

**Returns** `list[ZoneResult]` where each has `.zone_id`, `.count`, `.mean`, `.std_dev`, `.min_val`, `.max_val`, `.sum`

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
| `op` | `str` | required | `diff`, `ratio`, `anomaly_mean`, `anomaly_baseline`, `trend`, `mean`, `std`, `min`, `max` |
| `t0` | `int` | `0` | Index of the first scene (for `diff`, `ratio`) |
| `t1` | `int` | `-1` | Index of the second scene (-1 = last) |
| `baseline` | `str` | `"mean"` | Baseline method for anomaly operations |
| `time_values` | `list[float]` | `[]` | Timestamps for `trend` (defaults to 0, 1, 2, …) |

**`StackChain.temporal()` Returns** a `Chain` (can chain `algebra`, `clip`, `save_local`, etc.)

All scenes in the stack must have the same width, height, and CRS. Use `.reproject()` on each `Chain` before stacking if misaligned.

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
