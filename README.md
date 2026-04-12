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
chain = curaster.open("s3://my-bucket/data/scene.tif")   # S3 direct-read
chain = curaster.open("/vsis3/my-bucket/data/scene.tif") # GDAL vsis3 URI
```

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Local file path or S3 URI (`s3://` or `/vsis3/`) |

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
curaster.open("s3://my-bucket/scenes/LC08_2024_scene.tif") \
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

curaster.open("s3://my-bucket/raw/sentinel2.tif") \
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
