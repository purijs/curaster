/**
 * @file vram_cache.h
 * @brief VRAM-resident decoded raster cache for accelerated repeated processing.
 *
 * Loaded once via Chain::persist(). Each physical band is stored as:
 *   - Linear cudaMalloc device memory  → used by the algebra kernel
 *   - A CUDA 2D array + two texture objects (bilinear & nearest) → used by the warp kernel
 *
 * All derived chains share the cache via shared_ptr, so persist() data
 * survives algebra()/clip()/reproject() operations and is freed only when
 * every derived chain goes out of scope.
 *
 * @throws std::runtime_error (from Chain::persist()) if the raster exceeds 80%
 *         of currently-free VRAM or if raster dimensions exceed the CUDA 2D
 *         texture limit on the active device.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

struct VramCache {
    int width     = 0;
    int height    = 0;
    int num_bands = 0;  ///< Number of physical bands stored (all bands in the file).

    // ── per-physical-band linear device memory ──────────────────────────────
    // Indexed as d_bands[physical_band][pixel_index], where
    //   pixel_index = row * width + col  (row-major, chunk-relative via offset).
    // Engine algebra path: cache->d_bands[band_slots[b]] + chunk_y0 * width
    std::vector<float*> d_bands;

    // ── per-physical-band CUDA 2D arrays ────────────────────────────────────
    // cudaMallocArray'd; one per band; content equals d_bands[b].
    // Used as backing storage for the texture objects below.
    std::vector<cudaArray_t> d_arrays;

    // ── texture objects bound to d_arrays ───────────────────────────────────
    // Two per band (bilinear / nearest); both reference the same d_arrays[b].
    // Engine warp path picks the appropriate device-side array based on
    // ctx.reproject_params.resampling.
    std::vector<cudaTextureObject_t> tex_bilinear;
    std::vector<cudaTextureObject_t> tex_nearest;

    // Device-side arrays of texture handles (cudaMalloc'd, num_bands entries).
    // Indexed by physical band.  The engine remaps to band-slot order per chunk.
    cudaTextureObject_t* d_tex_bilinear_dev = nullptr;
    cudaTextureObject_t* d_tex_nearest_dev  = nullptr;

    VramCache()                            = default;
    VramCache(const VramCache&)            = delete;
    VramCache& operator=(const VramCache&) = delete;

    ~VramCache() {
        for (auto t : tex_bilinear) { if (t) cudaDestroyTextureObject(t); }
        for (auto t : tex_nearest)  { if (t) cudaDestroyTextureObject(t); }
        for (auto a : d_arrays)     { if (a) cudaFreeArray(a); }
        for (auto p : d_bands)      { if (p) cudaFree(p); }
        if (d_tex_bilinear_dev) { cudaFree(d_tex_bilinear_dev); d_tex_bilinear_dev = nullptr; }
        if (d_tex_nearest_dev)  { cudaFree(d_tex_nearest_dev);  d_tex_nearest_dev  = nullptr; }
    }
};