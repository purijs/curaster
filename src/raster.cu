#include <stdlib.h>
#include <iostream>
#include "gdal_priv.h"
#include "cpl_conv.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <bits/stdc++.h>
#include <pybind11/pybind11.h>

__global__ void compute_ndvi(const float* red_band, const float* nir_band, float* ndvi_band, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ndvi_band[idx] = (nir_band[idx] - red_band[idx]) / (nir_band[idx] + red_band[idx] + 1e-6);
    }
}

void curaster(std::string& input_file, std::string& output_file, int& red_band_index, int& nir_band_index) {
    GDALAllRegister();
    GDALDataset* rasterfile = (GDALDataset*)GDALOpen(input_file.c_str(), GA_ReadOnly);

    GDALRasterBand* band_red = rasterfile->GetRasterBand(red_band_index);
    GDALRasterBand* band_nir = rasterfile->GetRasterBand(nir_band_index);

    int blockXSize = band_red->GetXSize(), blockYSize = band_red->GetYSize();
    std::vector<float>* buffer_red_pixels = new std::vector<float>(blockXSize * blockYSize);
    std::vector<float>* buffer_nir_pixels = new std::vector<float>(blockXSize * blockYSize);

    CPLErr err = band_red->RasterIO(
        GF_Read,
        0, 0,
        blockXSize, blockYSize,
        buffer_red_pixels->data(),
        blockXSize, blockYSize,
        GDT_Float32,
        0, 0
    );

    CPLErr err_nir = band_nir->RasterIO(
        GF_Read,
        0, 0,
        blockXSize, blockYSize,
        buffer_nir_pixels->data(),
        blockXSize, blockYSize,
        GDT_Float32,
        0, 0
    );

    float* gpu_buffer_red_pixels, *gpu_buffer_nir_pixels, *gpu_buffer_ndvi_pixels;
    
    cudaMalloc((void**)&gpu_buffer_red_pixels, buffer_red_pixels->size() * sizeof(float));
    cudaMemcpy(gpu_buffer_red_pixels, buffer_red_pixels->data(), buffer_red_pixels->size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_buffer_nir_pixels, buffer_nir_pixels->size() * sizeof(float));
    cudaMemcpy(gpu_buffer_nir_pixels, buffer_nir_pixels->data(), buffer_nir_pixels->size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_buffer_ndvi_pixels, buffer_red_pixels->size() * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (buffer_red_pixels->size() + threadsPerBlock - 1) / threadsPerBlock;

    compute_ndvi<<<blocksPerGrid, threadsPerBlock>>>(gpu_buffer_red_pixels, gpu_buffer_nir_pixels, gpu_buffer_ndvi_pixels, buffer_red_pixels->size());
    cudaDeviceSynchronize();

    std::vector<float>* buffer_ndvi_pixels = new std::vector<float>(blockXSize * blockYSize);
    cudaMemcpy(buffer_ndvi_pixels->data(), gpu_buffer_ndvi_pixels, buffer_ndvi_pixels->size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_buffer_red_pixels);
    cudaFree(gpu_buffer_nir_pixels);

    delete buffer_red_pixels;
    delete buffer_nir_pixels;

    cudaFree(gpu_buffer_ndvi_pixels);

    std::cout << "Writing NDVI to disk..." << std::endl;

    double geoTransform[6];
    rasterfile->GetGeoTransform(geoTransform);
    const char* projection = rasterfile->GetProjectionRef();

    GDALDriver* gtiffDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* outRaster = gtiffDriver->Create(
        output_file.c_str(),
        blockXSize,
        blockYSize,
        1,
        GDT_Float32,
        nullptr
    );

    if (outRaster != nullptr) {
        outRaster->SetGeoTransform(geoTransform);
        outRaster->SetProjection(projection);

        CPLErr writeErr = outRaster->GetRasterBand(1)->RasterIO(
            GF_Write,
            0, 0,
            blockXSize, blockYSize,
            buffer_ndvi_pixels->data(),
            blockXSize, blockYSize,
            GDT_Float32,
            0, 0
        );

        GDALClose(outRaster);
    }
    
    delete buffer_ndvi_pixels;
    GDALClose(rasterfile);
}

PYBIND11_MODULE(curaster, m) {
    m.def("ndvi", &curaster, "A function that computes NDVI from a raster file and saves the output.",
        pybind11::arg("input_file"),
        pybind11::arg("output_file"),
        pybind11::arg("red_band_index"),
        pybind11::arg("nir_band_index")
    );
}