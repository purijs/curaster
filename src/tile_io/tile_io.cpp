/**
 * @file tile_io.cpp
 * @brief Tile/strip decompression, float conversion, and canvas fill routines.
 */
#include "tile_io.h"
#include "../tiff/decompression.h"

#include <algorithm>
#include <cstring>

// ─── decompress_tile_to_float ─────────────────────────────────────────────────

std::vector<float> decompress_tile_to_float(
    const uint8_t*     raw_bytes,
    size_t             raw_byte_count,
    int                block_width,
    int                block_height,
    int                samples_per_pixel,
    const std::string& compression,
    int                predictor,
    int                data_type) {

    std::vector<uint8_t> decoded;

    // ── Step 1: Decompress the raw block bytes ─────────────────────────────
    if (compression == "DEFLATE" || compression == "LZW") {
        if (!deflate_decompress(raw_bytes, raw_byte_count, decoded)) {
            decoded.assign(raw_bytes, raw_bytes + raw_byte_count);
        }
    } else if (compression == "ZSTD") {
        if (!zstd_decompress(raw_bytes, raw_byte_count, decoded)) {
            decoded.assign(raw_bytes, raw_bytes + raw_byte_count);
        }
    } else if (compression == "PACKBITS") {
        packbits_decompress(raw_bytes, raw_byte_count, decoded);
    } else {
        // NONE or unknown: treat as already uncompressed.
        decoded.assign(raw_bytes, raw_bytes + raw_byte_count);
    }

    // ── Step 2: Reverse the TIFF differencing predictor ───────────────────
    // Predictor 2 = horizontal differencing for integer data (UInt16).
    // Predictor 3 = floating-point differencing (Float32).
    if (predictor == 2 && data_type == 2) {
        unpredict_horizontal_u16(decoded.data(), block_width, block_height, samples_per_pixel);
    }
    if (predictor == 3 && data_type == 6) {
        unpredict_horizontal_f32(decoded.data(), block_width, block_height, samples_per_pixel);
    }

    // ── Step 3: Convert to float32 ────────────────────────────────────────
    size_t total_samples = static_cast<size_t>(block_width) * block_height * samples_per_pixel;
    std::vector<float> float_pixels(total_samples);

    if (data_type == 2) {
        // UInt16 → Float32 via AVX2-accelerated path.
        convert_u16_to_f32_avx2(
            reinterpret_cast<const uint16_t*>(decoded.data()),
            float_pixels.data(),
            total_samples);
    } else {
        // Float32: direct byte copy.
        memcpy(float_pixels.data(), decoded.data(), total_samples * sizeof(float));
    }

    return float_pixels;
}

// ─── extract_tile_bands ──────────────────────────────────────────────────────

void extract_tile_bands(
    const float*               tile_pixels,
    int                        tile_row,
    int                        tile_col,
    int                        chunk_y0,
    int                        chunk_height,
    int                        image_width,
    int                        block_width,
    int                        block_height,
    int                        samples_per_pixel,
    bool                       is_pixel_interleaved,
    const std::vector<int>&    band_slots,
    const std::vector<float*>& host_band_ptrs,
    int                        band_plane_index) {

    // Absolute row/col of the tile's top-left corner in image space.
    int tile_start_row = tile_row * block_height;
    int tile_start_col = tile_col * block_width;

    // Local row range inside the tile that overlaps the current chunk window.
    int local_row_first = std::max(0, chunk_y0 - tile_start_row);
    int local_row_last  = std::min(block_height - 1, chunk_y0 + chunk_height - 1 - tile_start_row);
    int local_col_last  = std::min(block_width  - 1, image_width - tile_start_col - 1);

    for (int local_row = local_row_first; local_row <= local_row_last; ++local_row) {
        int chunk_row  = tile_start_row + local_row - chunk_y0;  // Row in chunk output
        int image_col0 = tile_start_col;                          // Starting image column

        for (int local_col = 0; local_col <= local_col_last; ++local_col) {
            int image_col = image_col0 + local_col;

            if (is_pixel_interleaved) {
                // Pixel-interleaved: all bands share a stride of samples_per_pixel.
                size_t pixel_offset = static_cast<size_t>(local_row) * block_width + local_col;
                for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                    host_band_ptrs[slot][static_cast<size_t>(chunk_row) * image_width + image_col]
                        = tile_pixels[pixel_offset * samples_per_pixel + band_slots[slot]];
                }
            } else {
                // Band-planar: each tile contains only one band (band_plane_index).
                for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                    if (band_slots[slot] == band_plane_index) {
                        host_band_ptrs[slot][static_cast<size_t>(chunk_row) * image_width + image_col]
                            = tile_pixels[static_cast<size_t>(local_row) * block_width + local_col];
                        break;
                    }
                }
            }
        }
    }
}

// ─── fill_canvas_from_tile ───────────────────────────────────────────────────

void fill_canvas_from_tile(
    const float*               tile_pixels,
    int                        tile_start_row,
    int                        tile_start_col,
    int                        block_width,
    int                        block_height,
    int                        samples_per_pixel,
    bool                       is_pixel_interleaved,
    const std::vector<int>&    band_slots,
    const std::vector<float*>& canvas_band_ptrs,
    const SrcBBox&             canvas_bbox,
    int                        band_plane_index) {

    // Compute image-space overlap between the tile and the canvas bbox.
    int write_row_first = std::max(tile_start_row, canvas_bbox.y0);
    int write_row_last  = std::min(tile_start_row + block_height - 1,
                                   canvas_bbox.y0 + canvas_bbox.h  - 1);
    int write_col_first = std::max(tile_start_col, canvas_bbox.x0);
    int write_col_last  = std::min(tile_start_col + block_width  - 1,
                                   canvas_bbox.x0 + canvas_bbox.w - 1);

    if (write_row_first > write_row_last || write_col_first > write_col_last) {
        return; // No overlap.
    }

    for (int image_row = write_row_first; image_row <= write_row_last; ++image_row) {
        int local_tile_row    = image_row - tile_start_row;  // Row within the tile
        int canvas_output_row = image_row - canvas_bbox.y0;  // Row within the canvas

        for (int image_col = write_col_first; image_col <= write_col_last; ++image_col) {
            int local_tile_col    = image_col - tile_start_col;
            int canvas_output_col = image_col - canvas_bbox.x0;

            if (is_pixel_interleaved) {
                size_t pixel_offset = static_cast<size_t>(local_tile_row) * block_width
                                    + local_tile_col;
                for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                    canvas_band_ptrs[slot][
                        static_cast<size_t>(canvas_output_row) * canvas_bbox.w + canvas_output_col]
                        = tile_pixels[pixel_offset * samples_per_pixel + band_slots[slot]];
                }
            } else {
                for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                    if (band_slots[slot] == band_plane_index) {
                        canvas_band_ptrs[slot][
                            static_cast<size_t>(canvas_output_row) * canvas_bbox.w + canvas_output_col]
                            = tile_pixels[static_cast<size_t>(local_tile_row) * block_width
                                        + local_tile_col];
                        break;
                    }
                }
            }
        }
    }
}

// ─── fill_canvas_from_strip ──────────────────────────────────────────────────

void fill_canvas_from_strip(
    const float*               strip_pixels,
    int                        strip_first_row,
    int                        strip_num_rows,
    int                        source_full_width,
    int                        samples_per_pixel,
    bool                       is_pixel_interleaved,
    const std::vector<int>&    band_slots,
    const std::vector<float*>& canvas_band_ptrs,
    const SrcBBox&             canvas_bbox,
    int                        band_plane_index) {

    int overlap_row_first = std::max(strip_first_row, canvas_bbox.y0);
    int overlap_row_last  = std::min(strip_first_row + strip_num_rows - 1,
                                     canvas_bbox.y0 + canvas_bbox.h - 1);

    if (overlap_row_first > overlap_row_last) { return; }

    int canvas_col_first = canvas_bbox.x0;
    int canvas_col_last  = canvas_bbox.x0 + canvas_bbox.w - 1;

    for (int image_row = overlap_row_first; image_row <= overlap_row_last; ++image_row) {
        int strip_local_row   = image_row - strip_first_row;
        int canvas_output_row = image_row - canvas_bbox.y0;

        if (is_pixel_interleaved) {
            const float* strip_row = strip_pixels
                + static_cast<size_t>(strip_local_row) * source_full_width * samples_per_pixel;

            for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                int   band_channel = band_slots[slot];
                float* dest_row    = canvas_band_ptrs[slot]
                                   + static_cast<size_t>(canvas_output_row) * canvas_bbox.w;

                for (int col = canvas_col_first; col <= canvas_col_last; ++col) {
                    dest_row[col - canvas_col_first] =
                        strip_row[static_cast<size_t>(col) * samples_per_pixel + band_channel];
                }
            }
        } else {
            for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                if (band_slots[slot] == band_plane_index) {
                    const float* strip_row = strip_pixels
                        + static_cast<size_t>(strip_local_row) * source_full_width;
                    memcpy(
                        canvas_band_ptrs[slot]
                            + static_cast<size_t>(canvas_output_row) * canvas_bbox.w,
                        strip_row + canvas_col_first,
                        static_cast<size_t>(canvas_bbox.w) * sizeof(float));
                    break;
                }
            }
        }
    }
}
