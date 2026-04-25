/**
 * @file tile_io.h
 * @brief Tile and strip decompression, pixel extraction, and canvas filling.
 *
 * These functions form the bridge between raw compressed byte buffers (from
 * either GDAL or the S3 direct-read path) and the per-band float host buffers
 * consumed by the GPU algebra and warp kernels.
 */
#pragma once

#include <cstdint>  
#include <string>
#include <vector>
#include "../../include/types.h"  

/**
 * @brief Decompress one tile / strip block and return it as float32 pixels.
 *
 * Applies decompression (DEFLATE/LZW, ZSTD, or PACKBITS) then reverses
 * the TIFF differencing predictor if required, then converts to float32.
 *
 * @param raw_bytes        Pointer to the compressed block data.
 * @param raw_byte_count   Length of the compressed block in bytes.
 * @param block_width      Width of the block in pixels.
 * @param block_height     Height of the block in pixels.
 * @param samples_per_pixel Full samples-per-pixel count used in the compressed stream.
 * @param compression      Compression codec string ("DEFLATE", "ZSTD", etc.).
 * @param predictor        TIFF predictor tag (1, 2, or 3).
 * @param data_type        GDAL data type (2 = UInt16, 6 = Float32).
 * @return                 Float32 pixel values in row-major order.
 */
std::vector<float> decompress_tile_to_float(
    const uint8_t*     raw_bytes,
    size_t             raw_byte_count,
    int                block_width,
    int                block_height,
    int                samples_per_pixel,
    const std::string& compression,
    int                predictor,
    int                data_type);

/**
 * @brief Copy decompressed tile pixels into the caller's per-band output buffers.
 *
 * Handles both PIXEL-interleaved and BAND-planar layouts.
 *
 * @param tile_pixels      Decompressed float data (from decompress_tile_to_float).
 * @param tile_row         Tile row index.
 * @param tile_col         Tile column index.
 * @param chunk_y0         First row of the current processing chunk in image coords.
 * @param chunk_height     Height of the processing chunk in rows.
 * @param image_width      Full image width in pixels.
 * @param block_width      Tile block width.
 * @param block_height     Tile block height.
 * @param samples_per_pixel Samples per pixel in the tile stream (fi.spp or 1).
 * @param is_pixel_interleaved True if the tile is pixel-interleaved.
 * @param band_slots       Virtual→physical band slot mapping.
 * @param host_band_ptrs   Destination host pointers, one per band slot.
 * @param band_plane_index Physical band index; -1 for pixel-interleaved tiles.
 */
void extract_tile_bands(
    const float*              tile_pixels,
    int                       tile_row,
    int                       tile_col,
    int                       chunk_y0,
    int                       chunk_height,
    int                       image_width,
    int                       block_width,
    int                       block_height,
    int                       samples_per_pixel,
    bool                      is_pixel_interleaved,
    const std::vector<int>&   band_slots,
    const std::vector<float*>& host_band_ptrs,
    int                       band_plane_index = -1);

/**
 * @brief Copy decompressed tile pixels into a warp source canvas.
 *
 * Same as extract_tile_bands but writes into a bounded canvas window
 * (SrcBBox) instead of a full-width chunk buffer.
 */
void fill_canvas_from_tile(
    const float*              tile_pixels,
    int                       tile_start_row,
    int                       tile_start_col,
    int                       block_width,
    int                       block_height,
    int                       samples_per_pixel,
    bool                      is_pixel_interleaved,
    const std::vector<int>&   band_slots,
    const std::vector<float*>& canvas_band_ptrs,
    const SrcBBox&            canvas_bbox,
    int                       band_plane_index = -1);

/**
 * @brief Copy decompressed strip pixels into a warp source canvas.
 *
 * Equivalent of fill_canvas_from_tile for strip-layout files.
 */
void fill_canvas_from_strip(
    const float*              strip_pixels,
    int                       strip_first_row,
    int                       strip_num_rows,
    int                       source_full_width,
    int                       samples_per_pixel,
    bool                      is_pixel_interleaved,
    const std::vector<int>&   band_slots,
    const std::vector<float*>& canvas_band_ptrs,
    const SrcBBox&            canvas_bbox,
    int                       band_plane_index = -1);
