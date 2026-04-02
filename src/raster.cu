#include <stdlib.h>
#include <iostream>
#include "gdal_priv.h"
#include "cpl_conv.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include "cpl_progress.h"

const int threadsPerBlock = 256;

__global__ void compute_ndvi(const float* red_band, const float* nir_band, float* ndvi_band, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ndvi_band[idx] = (nir_band[idx] - red_band[idx]) / (nir_band[idx] + red_band[idx] + 1e-6);
    }
}

void curaster(std::string& input_file, std::string& output_file, int& red_band_index, int& nir_band_index, int& chunk_size, bool& verbose) {

    const int TILE_HEIGHT = chunk_size;

    GDALAllRegister();
    GDALDataset* rasterfile = (GDALDataset*)GDALOpen(input_file.c_str(), GA_ReadOnly);

    GDALRasterBand* band_red = rasterfile->GetRasterBand(red_band_index);
    GDALRasterBand* band_nir = rasterfile->GetRasterBand(nir_band_index);

    int blockXSize = band_red->GetXSize(), blockYSize = band_red->GetYSize();
    std::vector<float>* buffer_red_pixels = new std::vector<float>(TILE_HEIGHT * blockXSize);
    std::vector<float>* buffer_nir_pixels = new std::vector<float>(TILE_HEIGHT * blockXSize);
    std::vector<float>* buffer_ndvi_pixels = new std::vector<float>(TILE_HEIGHT * blockXSize);

    float* gpu_buffer_red_pixels, *gpu_buffer_nir_pixels, *gpu_buffer_ndvi_pixels;

    cudaMalloc((void**)&gpu_buffer_red_pixels, TILE_HEIGHT * blockXSize * sizeof(float));
    cudaMalloc((void**)&gpu_buffer_nir_pixels, TILE_HEIGHT * blockXSize * sizeof(float));
    cudaMalloc((void**)&gpu_buffer_ndvi_pixels, TILE_HEIGHT * blockXSize * sizeof(float));

    double geoTransform[6];
    rasterfile->GetGeoTransform(geoTransform);
    const char* projection = rasterfile->GetProjectionRef();

    GDALDriver* gtiffDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* outRaster = gtiffDriver->Create(
        output_file.c_str(), blockXSize, blockYSize, 1, GDT_Float32, nullptr
    );
    outRaster->SetGeoTransform(geoTransform);
    outRaster->SetProjection(projection);
    GDALRasterBand* outBand = outRaster->GetRasterBand(1);

    std::cout << "Writing NDVI to disk..." << std::endl;

    double progress = -1.0;
    if (verbose) {
        GDALTermProgress(0.0, nullptr, &progress);
    }

    for (int y_offset = 0; y_offset < blockYSize; y_offset += TILE_HEIGHT) {
        int current_tile_height = std::min(TILE_HEIGHT, blockYSize - y_offset);
        size_t current_pixels = current_tile_height * blockXSize;

        band_red->RasterIO(
            GF_Read,
            0, y_offset,
            blockXSize, current_tile_height,
            buffer_red_pixels->data(),
            blockXSize, current_tile_height,
            GDT_Float32,
            0, 0
        );

        band_nir->RasterIO(
            GF_Read,
            0, y_offset,
            blockXSize, current_tile_height,
            buffer_nir_pixels->data(),
            blockXSize, current_tile_height,
            GDT_Float32,
            0, 0
        );
    
        cudaMemcpy(gpu_buffer_red_pixels, buffer_red_pixels->data(), current_pixels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_buffer_nir_pixels, buffer_nir_pixels->data(), current_pixels * sizeof(float), cudaMemcpyHostToDevice);

        const int blocksPerGrid = (current_pixels + threadsPerBlock - 1) / threadsPerBlock;
        
        compute_ndvi<<<blocksPerGrid, threadsPerBlock>>>(gpu_buffer_red_pixels, gpu_buffer_nir_pixels, gpu_buffer_ndvi_pixels, current_pixels);
        cudaDeviceSynchronize();

        cudaMemcpy(buffer_ndvi_pixels->data(), gpu_buffer_ndvi_pixels, current_pixels * sizeof(float), cudaMemcpyDeviceToHost);

        outRaster->GetRasterBand(1)->RasterIO(
            GF_Write, 0, y_offset, blockXSize, current_tile_height,
            buffer_ndvi_pixels->data(), blockXSize, current_tile_height, GDT_Float32, 0, 0
        );

        if (verbose) {
            double fraction_complete = (double)(y_offset + current_tile_height) / blockYSize;
            GDALTermProgress(fraction_complete, nullptr, &progress);
        }

    }

    cudaFree(gpu_buffer_red_pixels);
    cudaFree(gpu_buffer_nir_pixels);
    cudaFree(gpu_buffer_ndvi_pixels);

    delete buffer_red_pixels;
    delete buffer_nir_pixels;
    delete buffer_ndvi_pixels;

    GDALClose(outRaster);
    GDALClose(rasterfile);
}

PYBIND11_MODULE(curaster, m) {

    m.doc() = "CUDA-accelerated Raster Algebra Calculator.";

    pybind11::module_ os = pybind11::module_::import("os");
    std::string module_dir = os.attr("path").attr("dirname")(m.attr("__file__")).cast<std::string>();
    
    std::string proj_path = module_dir + "/proj_data";
    std::string gdal_path = module_dir + "/gdal_data";
    
    os.attr("environ")["PROJ_LIB"] = proj_path;
    os.attr("environ")["PROJ_DATA"] = proj_path;
    os.attr("environ")["GDAL_DATA"] = gdal_path;

    CPLSetConfigOption("PROJ_LIB", proj_path.c_str());
    CPLSetConfigOption("PROJ_DATA", proj_path.c_str());
    CPLSetConfigOption("GDAL_DATA", gdal_path.c_str());

    m.def("ndvi", &curaster, R"pbdoc(
Computes the Normalized Difference Vegetation Index (NDVI) from a raster file.

This function reads the specified red and near-infrared (NIR) bands from the input
raster, processes the NDVI on the GPU, and writes the results directly to disk.

Args:
    input_file (str): Path to the input GeoTIFF file.
    output_file (str): Path where the resulting NDVI GeoTIFF will be saved.
    red_band_index (int): The index of the Red band.
    nir_band_index (int): The index of the Near-Infrared band.
    chunk_size (int, optional): The height of the chunk to process in memory at once. Defaults to 256.
    verbose (bool, optional): If True, displays a progress bar. Defaults to True.
)pbdoc",
        pybind11::arg("input_file"),
        pybind11::arg("output_file"),
        pybind11::arg("red_band_index"),
        pybind11::arg("nir_band_index"),
        pybind11::arg("chunk_size") = 256,
        pybind11::arg("verbose") = true
    );
}
