# cuRaster

Curaster is a high-performance Python module written in C++ and CUDA. It is designed to compute the Normalized Difference Vegetation Index (NDVI) on massive raster datasets using GPU.

## Installation

Install the precompiled wheel via pip:

```bash
pip install curaster
```

## Requirements (for building from source)

To build and run this module from source, your system needs:

- NVIDIA GPU with CUDA Toolkit installed (`nvcc`)
- GDAL C++ library (`libgdal-dev`)
- CMake 3.18+
- pybind11 (`pip install pybind11`)
- A C++17-compatible compiler

## Usage

Once installed, simply import it into Python:

```python
import curaster

curaster.ndvi(
    input_file="data/massive_satellite_image.tif",
    output_file="data/output_ndvi.tif",
    red_band_index=3,
    nir_band_index=4,
    chunk_size=256,    # Align this with your TIFF's BLOCKYSIZE for max speed
    verbose=True
)
```

## API Reference

### `curaster.ndvi(...)`

Calculates `(NIR - Red) / (NIR + Red)` and saves the result to a new single-band Float32 GeoTIFF.

| Argument         | Type   |  Default | Description                                                                 |
| ---------------- | ------ | -------: | --------------------------------------------------------------------------- |
| `input_file`     | `str`  | Required | Path to the input multispectral raster.                                     |
| `output_file`    | `str`  | Required | Path where the output NDVI raster will be saved.                            |
| `red_band_index` | `int`  | Required | The GDAL band number (1-indexed) containing Red data.                       |
| `nir_band_index` | `int`  | Required | The GDAL band number (1-indexed) containing NIR data.                       |
| `chunk_size`     | `int`  |    `256` | The number of horizontal rows to process at a time.                         |
| `verbose`        | `bool` |    `True`| If true, prints a progress bar to the terminal.                             |

## Building from Source

To build from source using CMake:

```bash
mkdir build
cd build
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make
```

Alternatively, you can compile directly with `nvcc`:

```bash
nvcc -O3 -shared -std=c++17 -Xcompiler -fPIC \
    -o curaster$(python3-config --extension-suffix) src/raster.cu \
    $(python3 -m pybind11 --includes) \
    $(gdal-config --cflags) $(gdal-config --libs)
```

## License

See [LICENSE](LICENSE) file for details.
