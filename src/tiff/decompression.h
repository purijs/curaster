/**
 * @file decompression.h
 * @brief TIFF tile / strip decompression and differencing-predictor reversal.
 *
 * Supports DEFLATE/LZW, ZSTD, and PACKBITS compressed blocks plus the
 * horizontal-differencing (predictor 2) and floating-point (predictor 3)
 * un-predict passes required before the raw bytes can be cast to float.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>



/**
 * @brief Decompress a DEFLATE / LZW block.
 * @param compressed_data   Pointer to compressed input bytes.
 * @param compressed_length Number of compressed bytes.
 * @param out               Output buffer; resized to hold decompressed data.
 * @return true on success, false if zlib reports an error.
 */
bool deflate_decompress(const uint8_t* compressed_data, size_t compressed_length,
                        std::vector<uint8_t>& out);

/**
 * @brief Decompress a Zstandard block.
 * @param compressed_data   Pointer to compressed input bytes.
 * @param compressed_length Number of compressed bytes.
 * @param out               Output buffer; resized to hold decompressed data.
 * @return true on success, false if zstd reports an error.
 */
bool zstd_decompress(const uint8_t* compressed_data, size_t compressed_length,
                     std::vector<uint8_t>& out);

/**
 * @brief Decode a PackBits-compressed block (no-copy run-length scheme).
 * @param compressed_data   Pointer to compressed input bytes.
 * @param compressed_length Number of compressed bytes.
 * @param out               Cleared and filled with decoded bytes.
 */
void packbits_decompress(const uint8_t* compressed_data, size_t compressed_length,
                         std::vector<uint8_t>& out);



/**
 * @brief Reverse TIFF Predictor 2 (horizontal differencing) for UInt16 data.
 *
 * Reverses the row-by-row prefix sum that TIFF encoders apply before
 * deflating 16-bit integer tiles.
 *
 * @param data     In-place data buffer: rows × (width × samples_per_pixel) uint16.
 * @param width    Tile/strip width in pixels.
 * @param height   Tile/strip height in rows.
 * @param samples_per_pixel  Samples per pixel (1 for planar, fi.spp for pixel-IL).
 */
void unpredict_horizontal_u16(uint8_t* data, int width, int height, int samples_per_pixel);

/**
 * @brief Reverse TIFF Predictor 3 (floating-point differencing) for Float32 data.
 *
 * Reverses the byte-plane split + horizontal prefix sum that encoders apply
 * before deflating 32-bit float tiles.
 *
 * @param data     In-place data buffer: rows × (width × samples_per_pixel) float.
 * @param width    Tile/strip width in pixels.
 * @param height   Tile/strip height in rows.
 * @param samples_per_pixel  Samples per pixel.
 */
void unpredict_horizontal_f32(uint8_t* data, int width, int height, int samples_per_pixel);



/**
 * @brief Convert @p count UInt16 samples to float32 using AVX2 if available.
 *
 * Falls back to a scalar loop for the trailing elements that do not fill
 * a full SIMD vector.
 *
 * @param src   Source UInt16 array.
 * @param dst   Destination float array (must be at least @p count elements).
 * @param count Number of samples to convert.
 */
void convert_u16_to_f32_avx2(const uint16_t* src, float* dst, size_t count);
