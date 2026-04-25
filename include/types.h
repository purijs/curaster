/**
 * @file types.h
 * @brief Core raster data types shared across the curaster pipeline.
 *
 * Contains metadata descriptors, bounding-box types, resampling options,
 * and the in-memory result container that is returned to Python callers.
 */
#pragma once

#include <string>
#include <vector>


/**
 * @brief Pixel resampling algorithm used during reprojection.
 */
enum class ResampleMethod {
    NEAREST  = 0,
    BILINEAR = 1,
};


/**
 * @brief Metadata that describes a GeoTIFF source file.
 *
 * Populated once by tiff_metadata::get_file_info() and then passed read-only
 * throughout the read, decompress, and warp pipeline stages.
 */
struct FileInfo {
    
    int width  = 0;
    int height = 0;

    
    int  tile_width  = 512;
    int  tile_height = 512;
    bool is_tiled    = false;

    
    int data_type          = 0;
    int samples_per_pixel  = 1;
    int predictor          = 1;

    
    std::string interleave  = "BAND";
    std::string compression = "NONE";
    int         rows_per_strip = 1;

    
    double      geo_transform[6] = {0, 1, 0, 0, 0, -1};
    std::string projection;
    bool        is_big_tiff = false;

    
    std::vector<size_t> tile_offsets;
    std::vector<size_t> tile_lengths;
    std::vector<size_t> strip_offsets;
    std::vector<size_t> strip_lengths;
};


/**
 * @brief A rectangle in source-image pixel space, used to limit canvas reads.
 */
struct SrcBBox {
    int x0;
    int y0;
    int w;
    int h;
};


/**
 * @brief Holds a fully computed raster for in-memory access from Python.
 *
 * Returned by Chain::to_memory(). Data is in raster-scan order
 * (row-major, top-to-bottom), one float per pixel.
 */
struct RasterResult {
    int      width  = 0;
    int      height = 0;
    int      bands  = 1;
    
    FileInfo    file_info;
    std::string projection;
    double      geo_transform[6] = {0, 1, 0, 0, 0, -1};

    std::vector<float> data;

    /// Allocate storage for width × height × bands floats, zero-initialised.
    void allocate() {
        data.assign(static_cast<size_t>(width) * height * bands, 0.0f);
    }
};
