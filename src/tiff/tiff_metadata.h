/**
 * @file tiff_metadata.h
 * @brief GeoTIFF metadata retrieval and output dataset creation via GDAL.
 */
#pragma once

#include <string>
#include "../../include/types.h"  // FileInfo

/**
 * @brief Open a GeoTIFF file and extract its metadata into a FileInfo.
 *
 * Works for both local paths and GDAL virtual filesystem paths (e.g. /vsicurl/).
 * Populates tile/strip offsets and lengths from the raw TIFF IFD so the S3
 * direct-read path can fetch individual blocks without GDAL involvement.
 *
 * @param file_path  Path or GDAL virtual path to the source GeoTIFF.
 * @return           Populated FileInfo struct.
 * @throws std::runtime_error if the file cannot be opened.
 */
FileInfo get_file_info(const std::string& file_path);

/**
 * @brief Create a new tiled GeoTIFF output dataset.
 *
 * Always writes 512×512 tiled BigTIFF with DEFLATE compression and
 * the floating-point predictor (Predictor=3), regardless of the source
 * file's block shape.
 *
 * @param output_path  Local file path for the output TIFF.
 * @param info         Metadata (width, height, projection, geotransform) to apply.
 * @return             A GDALDataset* opened for writing (caller must GDALClose it).
 * @throws std::runtime_error if the dataset cannot be created.
 */
class GDALDataset;  // Forward declaration — avoid pulling in gdal_priv.h everywhere
GDALDataset* create_output_dataset(const std::string& output_path, const FileInfo& info);
