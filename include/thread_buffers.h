/**
 * @file thread_buffers.h
 * @brief Per-thread GPU/CPU buffer management for the curaster pipeline.
 *
 * Each OMP worker thread owns one ThreadBufs instance holding:
 *  - Pinned host memory for band data and output (accessible at zero-copy speed)
 *  - Matching device pointers obtained via cudaHostGetDevicePointer
 *  - Optional polygon-clip span tables
 *  - Warp (reprojection) resources — source canvas, textures, coarse grid
 *
 * Allocation is split into two phases:
 *  1. alloc()       — core algebra / mask buffers (always allocated)
 *  2. alloc_warp()  — reprojection canvas + texture resources (warp path only)
 *
 * Call free_all() once processing is complete to release all resources.
 */
#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "raster_core.h"  // GpuSpanRow, WARP_GRID_WIDTH/HEIGHT
#include "types.h"        // ResampleMethod
#include "cuda_utils.h"   // CUDA_CHECK

// ─── Per-thread GPU buffer pool ───────────────────────────────────────────────
struct ThreadBufs {

    // ── Core algebra buffers ──────────────────────────────────────────────────

    /// Contiguous pinned allocation that backs all per-band host slices.
    float* h_pinned_master = nullptr;

    /// Per-band host pointers into h_pinned_master (one slice per band slot).
    std::vector<float*> h_bands;

    /// Per-band device pointers obtained via cudaHostGetDevicePointer.
    std::vector<float*> d_bands;

    /// Pinned host output buffer — one float per pixel in the current chunk.
    float* h_output = nullptr;

    /// Zero-copy device pointer for h_output.
    float* d_output = nullptr;

    /// Device-side array of device-band pointers, passed directly to kernels.
    float** d_band_ptrs = nullptr;

    // ── Polygon clip span buffers ─────────────────────────────────────────────
    /// Pinned host span table — one GpuSpanRow per row of the current chunk.
    GpuSpanRow* h_clip_spans = nullptr;

    /// Zero-copy device pointer for h_clip_spans.
    GpuSpanRow* d_clip_spans = nullptr;

    // ── CUDA stream ───────────────────────────────────────────────────────────
    cudaStream_t cuda_stream{};

    // ── Warp / reprojection resources (alloc_warp path only) ─────────────────
    /// Pinned host canvas — all bands laid out contiguously (band-major).
    float* h_src_canvas = nullptr;

    /// Per-band pointers into h_src_canvas.
    std::vector<float*> h_src_bands;

    /// CUDA arrays backing each source-band texture.
    std::vector<cudaArray_t> src_arrays;

    /// Texture objects for each band (bilinear or nearest, depending on mode).
    std::vector<cudaTextureObject_t> src_texture_objects;

    /// Device array of texture object handles passed to the warp kernel.
    cudaTextureObject_t* d_texture_objects = nullptr;

    /// Device control-point X coordinates for the WARP_GRID_WIDTH×HEIGHT grid.
    float* d_coarse_x = nullptr;

    /// Device control-point Y coordinates for the WARP_GRID_WIDTH×HEIGHT grid.
    float* d_coarse_y = nullptr;

    /// Device warp output — layout: [band * warp_max_chunk_pixels + pixel_index].
    float* d_warp_output = nullptr;

    /// Device-side per-band pointers into d_warp_output.
    float** d_warp_band_ptrs = nullptr;

    /// Stride (in floats) between band planes in d_warp_output.
    size_t warp_max_chunk_pixels = 0;

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @brief Allocate core buffers for band data, output, and optional clip spans.
     *
     * @param band_byte_size  Bytes required for one band of the current chunk.
     * @param num_bands       Number of virtual band slots.
     * @param max_clip_rows   Rows of span data to allocate; 0 = no clip mask.
     */
    void alloc(size_t band_byte_size, int num_bands, int max_clip_rows = 0) {
        // Single mapped+write-combining allocation for all band host buffers.
        CUDA_CHECK(cudaHostAlloc(
            &h_pinned_master,
            band_byte_size * num_bands,
            cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(h_pinned_master, 0, band_byte_size * num_bands);

        CUDA_CHECK(cudaHostAlloc(&h_output, band_byte_size, cudaHostAllocMapped));

        // Slice the pinned master buffer into per-band views.
        h_bands.resize(num_bands);
        d_bands.resize(num_bands);
        for (int band = 0; band < num_bands; ++band) {
            float* host_slice = h_pinned_master + band * (band_byte_size / sizeof(float));
            float* device_ptr = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&device_ptr), host_slice, 0));
            h_bands[band] = host_slice;
            d_bands[band] = device_ptr;
        }

        // Zero-copy device pointer for the output buffer.
        {
            float* device_out = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&device_out), h_output, 0));
            d_output = device_out;
        }

        // Copy device band pointers to GPU memory so kernels can index them.
        CUDA_CHECK(cudaMalloc(&d_band_ptrs, num_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(
            d_band_ptrs, d_bands.data(),
            num_bands * sizeof(float*), cudaMemcpyHostToDevice));

        // Optional: pinned span table for polygon clip masks.
        if (max_clip_rows > 0) {
            CUDA_CHECK(cudaHostAlloc(
                &h_clip_spans,
                max_clip_rows * sizeof(GpuSpanRow),
                cudaHostAllocMapped));
            memset(h_clip_spans, 0, max_clip_rows * sizeof(GpuSpanRow));
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&d_clip_spans), h_clip_spans, 0));
        }

        CUDA_CHECK(cudaStreamCreate(&cuda_stream));
    }

    /**
     * @brief Allocate reprojection resources (source canvas + textures + coarse grid).
     *
     * Must be called after alloc(). Only used on the reprojection code path.
     *
     * @param max_canvas_width   Maximum source canvas width in pixels.
     * @param max_canvas_height  Maximum source canvas height in pixels.
     * @param max_dst_pixels     Maximum destination pixels per chunk.
     * @param num_bands          Number of bands.
     * @param resample_method    Texture filter mode (nearest or bilinear).
     */
    void alloc_warp(int max_canvas_width, int max_canvas_height,
                    size_t max_dst_pixels, int num_bands,
                    ResampleMethod resample_method) {
        size_t canvas_pixels = static_cast<size_t>(max_canvas_width) * max_canvas_height;

        // Pinned canvas for all bands — read by bind_src_canvas into CUDA arrays.
        CUDA_CHECK(cudaHostAlloc(
            &h_src_canvas,
            canvas_pixels * num_bands * sizeof(float),
            cudaHostAllocDefault));

        h_src_bands.resize(num_bands);
        for (int band = 0; band < num_bands; ++band) {
            h_src_bands[band] = h_src_canvas + band * canvas_pixels;
        }

        // Create one 2-D CUDA array + texture object per band.
        cudaChannelFormatDesc channel_format = cudaCreateChannelDesc<float>();
        src_arrays.resize(num_bands);
        src_texture_objects.resize(num_bands);

        for (int band = 0; band < num_bands; ++band) {
            CUDA_CHECK(cudaMallocArray(
                &src_arrays[band], &channel_format,
                max_canvas_width, max_canvas_height));

            struct cudaResourceDesc resource_desc{};
            resource_desc.resType               = cudaResourceTypeArray;
            resource_desc.res.array.array        = src_arrays[band];

            struct cudaTextureDesc texture_desc{};
            texture_desc.addressMode[0]  = cudaAddressModeClamp;
            texture_desc.addressMode[1]  = cudaAddressModeClamp;
            texture_desc.filterMode      = (resample_method == ResampleMethod::BILINEAR)
                                               ? cudaFilterModeLinear
                                               : cudaFilterModePoint;
            texture_desc.readMode        = cudaReadModeElementType;
            texture_desc.normalizedCoords = 0;

            CUDA_CHECK(cudaCreateTextureObject(
                &src_texture_objects[band], &resource_desc, &texture_desc, nullptr));
        }

        // Upload texture handles to device memory for the kernel.
        CUDA_CHECK(cudaMalloc(&d_texture_objects, num_bands * sizeof(cudaTextureObject_t)));
        CUDA_CHECK(cudaMemcpy(
            d_texture_objects, src_texture_objects.data(),
            num_bands * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

        // Coarse warp grid in device memory.
        CUDA_CHECK(cudaMalloc(&d_coarse_x, WARP_GRID_WIDTH * WARP_GRID_HEIGHT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_coarse_y, WARP_GRID_WIDTH * WARP_GRID_HEIGHT * sizeof(float)));

        // Output buffer for the warp kernel (band-major).
        warp_max_chunk_pixels = max_dst_pixels;
        CUDA_CHECK(cudaMalloc(&d_warp_output, max_dst_pixels * num_bands * sizeof(float)));

        std::vector<float*> warp_band_ptrs(num_bands);
        for (int band = 0; band < num_bands; ++band) {
            warp_band_ptrs[band] = d_warp_output + static_cast<size_t>(band) * max_dst_pixels;
        }

        CUDA_CHECK(cudaMalloc(&d_warp_band_ptrs, num_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(
            d_warp_band_ptrs, warp_band_ptrs.data(),
            num_bands * sizeof(float*), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Upload source canvas data into CUDA arrays, ready for texture sampling.
     *
     * @param num_bands      Number of bands to upload.
     * @param canvas_width   Actual canvas width for this chunk.
     * @param canvas_height  Actual canvas height for this chunk.
     * @param stream         CUDA stream to use for the async memcpy.
     */
    void bind_src_canvas(int num_bands, int canvas_width, int canvas_height,
                         cudaStream_t stream) {
        size_t row_bytes = static_cast<size_t>(canvas_width) * sizeof(float);
        for (int band = 0; band < num_bands; ++band) {
            CUDA_CHECK(cudaMemcpy2DToArrayAsync(
                src_arrays[band], 0, 0,
                h_src_bands[band], row_bytes,
                row_bytes, static_cast<size_t>(canvas_height),
                cudaMemcpyHostToDevice, stream));
        }
    }

    /// Release all warp resources (textures, arrays, coarse grid, warp output).
    void free_warp() {
        if (!h_src_canvas) { return; }
        cudaFreeHost(h_src_canvas);
        h_src_canvas = nullptr;

        for (auto& tex : src_texture_objects) { cudaDestroyTextureObject(tex); }
        for (auto& arr : src_arrays)          { cudaFreeArray(arr); }
        src_texture_objects.clear();
        src_arrays.clear();

        cudaFree(d_texture_objects); d_texture_objects = nullptr;
        cudaFree(d_coarse_x);       d_coarse_x        = nullptr;
        cudaFree(d_coarse_y);       d_coarse_y        = nullptr;
        cudaFree(d_warp_output);    d_warp_output     = nullptr;
        cudaFree(d_warp_band_ptrs); d_warp_band_ptrs  = nullptr;
    }

    /// Release all resources (warp + core buffers + stream).
    void free_all() {
        free_warp();
        if (h_pinned_master) { cudaFreeHost(h_pinned_master); }
        if (h_output)        { cudaFreeHost(h_output); }
        if (d_band_ptrs)     { cudaFree(d_band_ptrs); }
        if (h_clip_spans)    { cudaFreeHost(h_clip_spans); }
        cudaStreamDestroy(cuda_stream);
    }
};
