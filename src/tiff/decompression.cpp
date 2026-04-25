/**
 * @file decompression.cpp
 * @brief TIFF tile / strip decompression and predictor-reversal implementations.
 */
#include "decompression.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>  

#include <zlib.h>
#include <zstd.h>



bool deflate_decompress(const uint8_t* compressed_data,
                        size_t         compressed_length,
                        std::vector<uint8_t>& out) {
    
    out.resize(compressed_length * 4);

    z_stream zstream{};
    zstream.avail_in = static_cast<uInt>(compressed_length);
    zstream.next_in  = const_cast<Bytef*>(compressed_data);
    inflateInit(&zstream);

    for (;;) {
        zstream.avail_out = static_cast<uInt>(out.size());
        zstream.next_out  = reinterpret_cast<Bytef*>(out.data());

        int result = inflate(&zstream, Z_FINISH);

        if (result == Z_STREAM_END) {
            out.resize(zstream.total_out);
            break;
        }
        if (result != Z_BUF_ERROR && result != Z_OK) {
            inflateEnd(&zstream);
            return false;
        }
        
        out.resize(out.size() * 2);
    }

    inflateEnd(&zstream);
    return true;
}



bool zstd_decompress(const uint8_t* compressed_data,
                     size_t         compressed_length,
                     std::vector<uint8_t>& out) {
    
    size_t decompressed_size = ZSTD_getFrameContentSize(compressed_data, compressed_length);

    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR
     || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        decompressed_size = compressed_length * 4; 
    }

    out.resize(decompressed_size);

    size_t result = ZSTD_decompress(
        out.data(), out.size(),
        compressed_data, compressed_length);

    if (ZSTD_isError(result)) {
        return false;
    }
    out.resize(result);
    return true;
}



void packbits_decompress(const uint8_t* compressed_data,
                         size_t         compressed_length,
                         std::vector<uint8_t>& out) {
    out.clear();
    size_t input_pos = 0;

    while (input_pos < compressed_length) {
        int8_t header = static_cast<int8_t>(compressed_data[input_pos++]);

        if (header >= 0) {
            
            int literal_count = header + 1;
            for (int i = 0; i < literal_count; ++i) {
                out.push_back(compressed_data[input_pos++]);
            }
        } else if (header != -128) {
            
            int   repeat_count = 1 - header;
            uint8_t repeat_byte = compressed_data[input_pos++];
            for (int i = 0; i < repeat_count; ++i) {
                out.push_back(repeat_byte);
            }
            
        }
    }
}



void unpredict_horizontal_u16(uint8_t* data,
                               int width, int height, int samples_per_pixel) {
    
    for (int row = 0; row < height; ++row) {
        auto* row_ptr = reinterpret_cast<uint16_t*>(
            data + static_cast<size_t>(row) * width * samples_per_pixel * 2);
        for (int col = 1; col < width * samples_per_pixel; ++col) {
            row_ptr[col] += row_ptr[col - 1];
        }
    }
}

void unpredict_horizontal_f32(uint8_t* data,
                               int width, int height, int samples_per_pixel) {
    
    
    size_t row_bytes = static_cast<size_t>(width) * samples_per_pixel * 4;

    for (int row = 0; row < height; ++row) {
        uint8_t* row_base = data + static_cast<size_t>(row) * row_bytes;
        int      plane_stride = width * samples_per_pixel;

        
        uint8_t* plane[4] = {
            row_base + 0             * plane_stride,
            row_base + 1             * plane_stride,
            row_base + 2             * plane_stride,
            row_base + 3             * plane_stride,
        };

        
        for (int col = 1; col < width * samples_per_pixel; ++col) {
            plane[0][col] += plane[0][col - 1];
            plane[1][col] += plane[1][col - 1];
            plane[2][col] += plane[2][col - 1];
            plane[3][col] += plane[3][col - 1];
        }

        
        std::vector<uint8_t> interleaved(row_bytes);
        for (int col = 0; col < width * samples_per_pixel; ++col) {
            interleaved[col * 4 + 0] = plane[0][col];
            interleaved[col * 4 + 1] = plane[1][col];
            interleaved[col * 4 + 2] = plane[2][col];
            interleaved[col * 4 + 3] = plane[3][col];
        }
        memcpy(row_base, interleaved.data(), row_bytes);
    }
}



void convert_u16_to_f32_avx2(const uint16_t* src, float* dst, size_t count) {
    size_t vec_idx = 0;

    
    for (; vec_idx + 16 <= count; vec_idx += 16) {
        __m256i low_8  = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(src + vec_idx)));
        __m256i high_8 = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(src + vec_idx + 8)));

        _mm256_storeu_ps(dst + vec_idx,     _mm256_cvtepi32_ps(low_8));
        _mm256_storeu_ps(dst + vec_idx + 8, _mm256_cvtepi32_ps(high_8));
    }

    
    for (; vec_idx < count; ++vec_idx) {
        dst[vec_idx] = static_cast<float>(src[vec_idx]);
    }
}
