/**
 * @file raster_cache.h
 * @brief VRAM-resident raster cache enabling zero-I/O repeated GPU operations.
 *
 * Created by Chain::persist().  Holds all bands of the source raster pre-decoded
 * in GPU device memory so that subsequent engine calls skip S3/disk I/O entirely.
 *
 * Two representations per band are maintained:
 *   d_bands     – contiguous float* device memory; used by engine_ex (algebra/clip).
 *   src_arrays  – cudaArray_t with bilinear texture objects; used by engine_reproject
 *                 so that tex2D samples the full image at absolute source coordinates
 *                 without any canvas upload.
 */
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "types.h"

struct VramCache {
    int   width     = 0;
    int   height    = 0;
    int   num_bands = 0;
    FileInfo file_info;

    /// Linear device memory, one cudaMalloc per band (width * height * sizeof(float)).
    std::vector<float*>              d_bands;

    /// CUDA 2D arrays (one per band) — supports hardware bilinear tex2D sampling.
    std::vector<cudaArray_t>         src_arrays;

    /// Texture objects bound to src_arrays (bilinear, clamp, non-normalised coords).
    std::vector<cudaTextureObject_t> tex_objects;

    /// Device-side pointer array (cudaMalloc) for passing tex_objects to warp kernel.
    cudaTextureObject_t*             d_tex_array = nullptr;

    VramCache() = default;
    VramCache(const VramCache&) = delete;
    VramCache& operator=(const VramCache&) = delete;

    ~VramCache() { release(); }

    void release() noexcept {
        for (auto tex : tex_objects)  cudaDestroyTextureObject(tex);
        for (auto arr : src_arrays)   cudaFreeArray(arr);
        for (auto ptr : d_bands)      cudaFree(ptr);
        if (d_tex_array) { cudaFree(d_tex_array); d_tex_array = nullptr; }
        tex_objects.clear();
        src_arrays.clear();
        d_bands.clear();
        width = height = num_bands = 0;
    }
};