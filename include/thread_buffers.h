#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "raster_core.h"
#include "types.h"
#include "cuda_utils.h"
#include "pinned_arena.h"

struct ThreadBufs {

    float* h_pinned_master = nullptr;
    std::vector<float*> h_bands;
    std::vector<float*> d_bands;
    float* h_output   = nullptr;
    float* d_output   = nullptr;
    float** d_band_ptrs = nullptr;

    GpuSpanRow* h_clip_spans = nullptr;
    GpuSpanRow* d_clip_spans = nullptr;

    cudaStream_t cuda_stream{};

    float* h_src_canvas = nullptr;
    std::vector<float*> h_src_bands;
    std::vector<cudaArray_t> src_arrays;
    std::vector<cudaTextureObject_t> src_texture_objects;
    cudaTextureObject_t* d_texture_objects = nullptr;
    float* d_coarse_x   = nullptr;
    float* d_coarse_y   = nullptr;
    float* d_warp_output = nullptr;
    float** d_warp_band_ptrs = nullptr;
    float* h_warp_multiband = nullptr;
    size_t warp_max_chunk_pixels = 0;

    float*  h_halo_master         = nullptr;
    float*  d_halo_device         = nullptr;
    float*  d_neighborhood_output = nullptr;
    float** d_neighborhood_band_ptrs = nullptr;
    size_t  halo_alloc_bytes      = 0;
    int     neighborhood_num_bands = 0;
    size_t  neighborhood_max_chunk_pixels = 0;

    uint16_t* h_zone_labels  = nullptr;
    uint16_t* d_zone_labels  = nullptr;
    int*      d_zone_count   = nullptr;
    float*    d_zone_sum     = nullptr;
    float*    d_zone_sum_sq  = nullptr;
    float*    d_zone_min_buf = nullptr;
    float*    d_zone_max_buf = nullptr;
    int       zonal_num_zones = 0;

    float*  d_stack_output    = nullptr;
    float** d_stack_band_ptrs = nullptr;
    int     stack_num_scenes  = 0;
    size_t  stack_max_pixels  = 0;

    void alloc(size_t band_byte_size, int num_bands, int max_clip_rows = 0) {
        CUDA_CHECK(cudaHostAlloc(
            &h_pinned_master,
            band_byte_size * num_bands,
            cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(h_pinned_master, 0, band_byte_size * num_bands);

        CUDA_CHECK(cudaHostAlloc(&h_output, band_byte_size, cudaHostAllocMapped));

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

        {
            float* device_out = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&device_out), h_output, 0));
            d_output = device_out;
        }

        CUDA_CHECK(cudaMalloc(&d_band_ptrs, num_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(
            d_band_ptrs, d_bands.data(),
            num_bands * sizeof(float*), cudaMemcpyHostToDevice));

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

    void alloc_from_arena(PinnedArena& arena, size_t band_byte_size,
                          int num_bands, int max_clip_rows = 0) {
        h_pinned_master = arena.alloc_typed<float>(
            band_byte_size * num_bands / sizeof(float));
        memset(h_pinned_master, 0, band_byte_size * num_bands);

        h_output = arena.alloc_typed<float>(band_byte_size / sizeof(float));
        memset(h_output, 0, band_byte_size);

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

        {
            float* device_out = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&device_out), h_output, 0));
            d_output = device_out;
        }

        CUDA_CHECK(cudaMalloc(&d_band_ptrs, num_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d_band_ptrs, d_bands.data(),
                              num_bands * sizeof(float*), cudaMemcpyHostToDevice));

        if (max_clip_rows > 0) {
            h_clip_spans = arena.alloc_typed<GpuSpanRow>(max_clip_rows);
            memset(h_clip_spans, 0, max_clip_rows * sizeof(GpuSpanRow));
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&d_clip_spans), h_clip_spans, 0));
        }

        if (!cuda_stream) {
            CUDA_CHECK(cudaStreamCreate(&cuda_stream));
        }
    }

    void alloc_halo_from_arena(PinnedArena& arena, size_t halo_bytes,
                               int num_out_bands, size_t max_chunk_pixels) {
        h_halo_master = arena.alloc_typed<float>(halo_bytes / sizeof(float));
        memset(h_halo_master, 0, halo_bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&d_halo_device), h_halo_master, 0));

        CUDA_CHECK(cudaMalloc(&d_neighborhood_output,
                              max_chunk_pixels * num_out_bands * sizeof(float)));

        std::vector<float*> ptrs(num_out_bands);
        for (int b = 0; b < num_out_bands; ++b) {
            ptrs[b] = d_neighborhood_output + static_cast<size_t>(b) * max_chunk_pixels;
        }
        CUDA_CHECK(cudaMalloc(&d_neighborhood_band_ptrs, num_out_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d_neighborhood_band_ptrs, ptrs.data(),
                              num_out_bands * sizeof(float*), cudaMemcpyHostToDevice));

        halo_alloc_bytes           = halo_bytes;
        neighborhood_num_bands     = num_out_bands;
        neighborhood_max_chunk_pixels = max_chunk_pixels;
    }

    void alloc_warp(int max_canvas_width, int max_canvas_height,
                    size_t max_dst_pixels, int num_bands,
                    ResampleMethod resample_method) {
        size_t canvas_pixels = static_cast<size_t>(max_canvas_width) * max_canvas_height;

        CUDA_CHECK(cudaHostAlloc(
            &h_src_canvas,
            canvas_pixels * num_bands * sizeof(float),
            cudaHostAllocDefault));

        h_src_bands.resize(num_bands);
        for (int band = 0; band < num_bands; ++band) {
            h_src_bands[band] = h_src_canvas + band * canvas_pixels;
        }

        cudaChannelFormatDesc channel_format = cudaCreateChannelDesc<float>();
        src_arrays.resize(num_bands);
        src_texture_objects.resize(num_bands);

        for (int band = 0; band < num_bands; ++band) {
            CUDA_CHECK(cudaMallocArray(
                &src_arrays[band], &channel_format,
                max_canvas_width, max_canvas_height));

            struct cudaResourceDesc resource_desc{};
            resource_desc.resType              = cudaResourceTypeArray;
            resource_desc.res.array.array       = src_arrays[band];

            struct cudaTextureDesc texture_desc{};
            texture_desc.addressMode[0] = cudaAddressModeClamp;
            texture_desc.addressMode[1] = cudaAddressModeClamp;
            texture_desc.filterMode     = (resample_method == ResampleMethod::BILINEAR)
                                              ? cudaFilterModeLinear
                                              : cudaFilterModePoint;
            texture_desc.readMode        = cudaReadModeElementType;
            texture_desc.normalizedCoords = 0;

            CUDA_CHECK(cudaCreateTextureObject(
                &src_texture_objects[band], &resource_desc, &texture_desc, nullptr));
        }

        CUDA_CHECK(cudaMalloc(&d_texture_objects, num_bands * sizeof(cudaTextureObject_t)));
        CUDA_CHECK(cudaMemcpy(
            d_texture_objects, src_texture_objects.data(),
            num_bands * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_coarse_x, WARP_GRID_WIDTH * WARP_GRID_HEIGHT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_coarse_y, WARP_GRID_WIDTH * WARP_GRID_HEIGHT * sizeof(float)));

        warp_max_chunk_pixels = max_dst_pixels;
        CUDA_CHECK(cudaMalloc(&d_warp_output, max_dst_pixels * num_bands * sizeof(float)));
        CUDA_CHECK(cudaHostAlloc(&h_warp_multiband,
            max_dst_pixels * num_bands * sizeof(float), cudaHostAllocDefault));

        std::vector<float*> warp_band_ptrs(num_bands);
        for (int band = 0; band < num_bands; ++band) {
            warp_band_ptrs[band] = d_warp_output + static_cast<size_t>(band) * max_dst_pixels;
        }

        CUDA_CHECK(cudaMalloc(&d_warp_band_ptrs, num_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(
            d_warp_band_ptrs, warp_band_ptrs.data(),
            num_bands * sizeof(float*), cudaMemcpyHostToDevice));
    }

    void alloc_halo(size_t halo_bytes, int num_out_bands, size_t max_chunk_pixels) {
        if (halo_bytes == halo_alloc_bytes
                && num_out_bands == neighborhood_num_bands
                && max_chunk_pixels == neighborhood_max_chunk_pixels) {
            return;
        }
        free_halo();
        CUDA_CHECK(cudaHostAlloc(&h_halo_master, halo_bytes,
                                 cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(h_halo_master, 0, halo_bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&d_halo_device), h_halo_master, 0));

        CUDA_CHECK(cudaMalloc(&d_neighborhood_output,
                              max_chunk_pixels * num_out_bands * sizeof(float)));

        std::vector<float*> ptrs(num_out_bands);
        for (int b = 0; b < num_out_bands; ++b) {
            ptrs[b] = d_neighborhood_output + static_cast<size_t>(b) * max_chunk_pixels;
        }
        CUDA_CHECK(cudaMalloc(&d_neighborhood_band_ptrs, num_out_bands * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d_neighborhood_band_ptrs, ptrs.data(),
                              num_out_bands * sizeof(float*), cudaMemcpyHostToDevice));

        halo_alloc_bytes          = halo_bytes;
        neighborhood_num_bands     = num_out_bands;
        neighborhood_max_chunk_pixels = max_chunk_pixels;
    }

    void free_halo() {
        if (h_halo_master)           { cudaFreeHost(h_halo_master); h_halo_master = nullptr; }
        if (d_neighborhood_output)   { cudaFree(d_neighborhood_output); d_neighborhood_output = nullptr; }
        if (d_neighborhood_band_ptrs){ cudaFree(d_neighborhood_band_ptrs); d_neighborhood_band_ptrs = nullptr; }
        d_halo_device = nullptr;
        halo_alloc_bytes = 0;
        neighborhood_num_bands = 0;
    }

    void alloc_zonal(int num_zones, int max_chunk_rows, int width) {
        if (zonal_num_zones == num_zones) { return; }
        free_zonal();
        size_t label_bytes = static_cast<size_t>(width) * max_chunk_rows * sizeof(uint16_t);
        CUDA_CHECK(cudaHostAlloc(&h_zone_labels, label_bytes, cudaHostAllocMapped));
        memset(h_zone_labels, 0, label_bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&d_zone_labels), h_zone_labels, 0));

        CUDA_CHECK(cudaMalloc(&d_zone_count,   (num_zones + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_zone_sum,     (num_zones + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_zone_sum_sq,  (num_zones + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_zone_min_buf, (num_zones + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_zone_max_buf, (num_zones + 1) * sizeof(float)));
        zonal_num_zones = num_zones;
    }

    void free_zonal() {
        if (h_zone_labels) { cudaFreeHost(h_zone_labels); h_zone_labels = nullptr; d_zone_labels = nullptr; }
        if (d_zone_count)   { cudaFree(d_zone_count);   d_zone_count   = nullptr; }
        if (d_zone_sum)     { cudaFree(d_zone_sum);     d_zone_sum     = nullptr; }
        if (d_zone_sum_sq)  { cudaFree(d_zone_sum_sq);  d_zone_sum_sq  = nullptr; }
        if (d_zone_min_buf) { cudaFree(d_zone_min_buf); d_zone_min_buf = nullptr; }
        if (d_zone_max_buf) { cudaFree(d_zone_max_buf); d_zone_max_buf = nullptr; }
        zonal_num_zones = 0;
    }

    void alloc_stack(int num_scenes, size_t max_pixels) {
        if (stack_num_scenes == num_scenes && stack_max_pixels == max_pixels) { return; }
        free_stack();
        CUDA_CHECK(cudaMalloc(&d_stack_output, num_scenes * max_pixels * sizeof(float)));
        std::vector<float*> ptrs(num_scenes);
        for (int s = 0; s < num_scenes; ++s) {
            ptrs[s] = d_stack_output + static_cast<size_t>(s) * max_pixels;
        }
        CUDA_CHECK(cudaMalloc(&d_stack_band_ptrs, num_scenes * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d_stack_band_ptrs, ptrs.data(),
                              num_scenes * sizeof(float*), cudaMemcpyHostToDevice));
        stack_num_scenes = num_scenes;
        stack_max_pixels = max_pixels;
    }

    void free_stack() {
        if (d_stack_output)    { cudaFree(d_stack_output);    d_stack_output    = nullptr; }
        if (d_stack_band_ptrs) { cudaFree(d_stack_band_ptrs); d_stack_band_ptrs = nullptr; }
        stack_num_scenes = 0;
        stack_max_pixels = 0;
    }

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
        if (h_warp_multiband) { cudaFreeHost(h_warp_multiband); h_warp_multiband = nullptr; }
    }

    void free_all() {
        free_warp();
        free_halo();
        free_zonal();
        free_stack();
        if (h_pinned_master) { cudaFreeHost(h_pinned_master); h_pinned_master = nullptr; }
        if (h_output)        { cudaFreeHost(h_output);        h_output        = nullptr; }
        if (d_band_ptrs)     { cudaFree(d_band_ptrs);         d_band_ptrs     = nullptr; }
        if (h_clip_spans)    { cudaFreeHost(h_clip_spans);    h_clip_spans    = nullptr; }
        cudaStreamDestroy(cuda_stream);
    }

    void free_device_only() {
        if (d_band_ptrs)             { cudaFree(d_band_ptrs);             d_band_ptrs             = nullptr; }
        if (d_neighborhood_output)   { cudaFree(d_neighborhood_output);   d_neighborhood_output   = nullptr; }
        if (d_neighborhood_band_ptrs){ cudaFree(d_neighborhood_band_ptrs);d_neighborhood_band_ptrs= nullptr; }
        free_zonal();
        free_stack();
    }
};
