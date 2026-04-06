# cuRaster

cuRaster is a high-performance Python module written in C++ and CUDA. It is designed to compute general raster algebra expressions on massive raster datasets using GPU acceleration.

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
- A C++20-compatible compiler

## Usage

Once installed, simply import it into Python:

```python
import curaster

curaster.compute(
    input_file="input.tif",
    output_file="output.tif",
    expression="(B4-B1)/(B4+B1)",
    verbose=True
)
```

## API Reference

### `curaster.compute(...)`

Computes raster algebra from a string expression on the GPU and saves the result to a new single-band Float32 GeoTIFF.

| Argument     | Type   |  Default | Description                                                                 |
| ------------ | ------ | -------: | --------------------------------------------------------------------------- |
| `input_file` | `str`  | Required | Path to the input multispectral raster                                     |
| `output_file`| `str`  | Required | Path where the output raster will be saved                                 |
| `expression` | `str`  | Required | The algebraic formula (e.g., "(B4-B1)/(B4+B1)"). Bands are referenced as B1, B2, etc. (1-indexed) |
| `verbose`    | `bool` |    `True`| If true, prints memory telemetry and a progress bar to the terminal      |

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
nvcc -O3 -shared -std=c++20 -Xcompiler -fPIC -Xcompiler -fopenmp --use_fast_math \
    -o curaster$(python3-config --extension-suffix) src/raster.cu \
    $(python3 -m pybind11 --includes) \
    $(gdal-config --cflags) $(gdal-config --libs)
```

## License

See [LICENSE](LICENSE) file for details.
