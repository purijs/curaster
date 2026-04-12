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

// ─── Resampling algorithm ─────────────────────────────────────────────────────
/**
 * @brief Pixel resampling algorithm used during reprojection.
 */
enum class ResampleMethod {
    NEAREST  = 0, ///< Nearest-neighbour — no interpolation, fastest
    BILINEAR = 1, ///< Bilinear interpolation — default, smoother results
};

// ─── Source raster metadata ───────────────────────────────────────────────────
/**
 * @brief Metadata that describes a GeoTIFF source file.
 *
 * Populated once by tiff_metadata::get_file_info() and then passed read-only
 * throughout the read, decompress, and warp pipeline stages.
 */
struct FileInfo {
    // ── Raster dimensions ─────────────────────────────────────────────────
    int width  = 0; ///< Image width  in pixels
    int height = 0; ///< Image height in pixels

    // ── Block / tile layout ───────────────────────────────────────────────
    int  tile_width  = 512;  ///< Block width  in pixels (equals image width for stripped files)
    int  tile_height = 512;  ///< Block height in pixels (equals 1 or rows_per_strip for strips)
    bool is_tiled    = false; ///< True → tiled TIFF, False → stripped TIFF

    // ── Pixel format ──────────────────────────────────────────────────────
    int data_type          = 0; ///< GDAL GDALDataType (GDT_Float32 = 6, GDT_UInt16 = 2)
    int samples_per_pixel  = 1; ///< Samples (bands) per pixel within one compressed block
    int predictor          = 1; ///< TIFF Predictor tag (1 = none, 2 = horizontal, 3 = float)

    // ── Storage layout ────────────────────────────────────────────────────
    std::string interleave  = "BAND";  ///< "BAND" (planar) or "PIXEL" (interleaved)
    std::string compression = "NONE";  ///< Codec: "DEFLATE", "LZW", "ZSTD", "PACKBITS", …
    int         rows_per_strip = 1;    ///< Rows per strip (strip layout only)

    // ── Geospatial reference ──────────────────────────────────────────────
    double      geo_transform[6] = {0, 1, 0, 0, 0, -1}; ///< GDAL 6-element affine geotransform
    std::string projection;   ///< WKT coordinate reference system string
    bool        is_big_tiff = false; ///< True if the file uses the BigTIFF (>4 GB) variant

    // ── Raw tile / strip byte extents (used on the S3 direct-read path) ──
    std::vector<size_t> tile_offsets;  ///< Byte offset of each tile in the file
    std::vector<size_t> tile_lengths;  ///< Compressed byte length of each tile
    std::vector<size_t> strip_offsets; ///< Byte offset of each strip in the file
    std::vector<size_t> strip_lengths; ///< Compressed byte length of each strip
};

// ─── Source bounding box (pixel coordinates) ─────────────────────────────────
/**
 * @brief A rectangle in source-image pixel space, used to limit canvas reads.
 */
struct SrcBBox {
    int x0; ///< Left   column (inclusive)
    int y0; ///< Top    row    (inclusive)
    int w;  ///< Width  in pixels
    int h;  ///< Height in pixels
};

// ─── In-memory processing result ─────────────────────────────────────────────
/**
 * @brief Holds a fully computed raster for in-memory access from Python.
 *
 * Returned by Chain::to_memory(). Data is in raster-scan order
 * (row-major, top-to-bottom), one float per pixel.
 */
struct RasterResult {
    int      width  = 0; ///< Output width  in pixels
    int      height = 0; ///< Output height in pixels

    FileInfo    file_info;                              ///< Metadata matching the output raster
    std::string projection;                             ///< WKT CRS (copied from file_info)
    double      geo_transform[6] = {0, 1, 0, 0, 0, -1}; ///< Output affine geotransform

    std::vector<float> data; ///< Pixel values in row-major order (width × height floats)

    /// Allocate storage for width × height floats, zero-initialised.
    void allocate() {
        data.assign(static_cast<size_t>(width) * height, 0.0f);
    }
};
