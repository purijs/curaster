#include "engine.h"
#include "engine_temporal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gdal_priv.h"
#include "cpl_progress.h"

#include "../../include/cuda_utils.h"
#include "../../include/raster_core.h"
#include "../../include/thread_buffers.h"
#include "../../include/pipeline.h"

#include "../tiff/tiff_metadata.h"



struct StackPoolKey {
    int    num_scenes;
    int    chunk_height;
    int    width;
    bool operator==(const StackPoolKey& o) const noexcept {
        return num_scenes == o.num_scenes && chunk_height == o.chunk_height && width == o.width;
    }
};

static ThreadBufs    g_stack_buf;
static StackPoolKey  g_stack_key{0, 0, 0};
static bool          g_stack_init = false;



static std::vector<float*> g_scene_bufs_b;     
static std::vector<float*> g_scene_bufs_b_dev; 
static float*              g_output_b     = nullptr;
static float*              g_output_b_dev = nullptr;
static cudaStream_t        g_stack_stream_b{};

static void stack_ensure_pool(int N, int chunk_height, int width) {
    StackPoolKey needed{N, chunk_height, width};
    if (g_stack_init && needed == g_stack_key) return;

    if (g_stack_init) {
        g_stack_buf.free_stack();
        g_stack_buf.free_all();
        for (float* p : g_scene_bufs_b) { if (p) cudaFreeHost(p); }
        g_scene_bufs_b.clear();
        g_scene_bufs_b_dev.clear();
        if (g_output_b) { cudaFreeHost(g_output_b); g_output_b = nullptr; }
        cudaStreamDestroy(g_stack_stream_b);
    }

    size_t per_scene_bytes = (size_t)width * chunk_height * sizeof(float);

    
    
    g_stack_buf.alloc(per_scene_bytes, 1, 0);  
    g_stack_buf.alloc_stack(N, (size_t)width * chunk_height);

    
    g_scene_bufs_b.resize(N, nullptr);
    g_scene_bufs_b_dev.resize(N, nullptr);
    for (int s = 0; s < N; ++s) {
        CUDA_CHECK(cudaHostAlloc(&g_scene_bufs_b[s], per_scene_bytes,
                                 cudaHostAllocDefault));
    }

    
    CUDA_CHECK(cudaHostAlloc(&g_output_b, per_scene_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&g_output_b_dev), g_output_b, 0));

    CUDA_CHECK(cudaStreamCreate(&g_stack_stream_b));
    g_stack_key  = needed;
    g_stack_init = true;
}

void release_stack_pool() {
    if (!g_stack_init) return;
    g_stack_buf.free_stack();
    g_stack_buf.free_all();
    for (float* p : g_scene_bufs_b) { if (p) cudaFreeHost(p); }
    g_scene_bufs_b.clear();
    g_scene_bufs_b_dev.clear();
    if (g_output_b) { cudaFreeHost(g_output_b); g_output_b = nullptr; }
    if (g_stack_stream_b) { cudaStreamDestroy(g_stack_stream_b); g_stack_stream_b = {}; }
    g_stack_key  = StackPoolKey{0, 0, 0};
    g_stack_init = false;
}



void run_engine_temporal(const std::vector<std::string>& scene_files,
                          PipelineCtx& ctx,
                          bool verbose)
{
    GDALAllRegister();
    release_warp_pool();
    release_halo_pool();
    release_zonal_pool();
    g_pinned_arena.release();
    init_ram_budget();

    int N = static_cast<int>(scene_files.size());
    if (N == 0) throw std::runtime_error("run_engine_temporal: empty scene list");

    FileInfo ref_info = get_file_info(scene_files[0]);
    int width  = ref_info.width;
    int height = ref_info.height;

    
    
    
    
    
    
    
    
    
    
    
    

    size_t free_vram = 0, tot_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &tot_vram));

    
    
    size_t bytes_per_row_ram = (size_t)width * (2 * N + 3) * sizeof(float);
    size_t max_rows_ram = static_cast<size_t>(g_pinned_budget * 0.60) /
                          bytes_per_row_ram / 2;

    size_t vram_per_row  = (size_t)N * width * sizeof(float);
    size_t max_rows_vram = (size_t)(free_vram * 0.85) / vram_per_row;

    size_t max_rows = std::min(max_rows_ram, max_rows_vram);

    int chunk_height = ref_info.tile_height;
    while (chunk_height + ref_info.tile_height <= (int)max_rows &&
           chunk_height + ref_info.tile_height <= height) {
        chunk_height += ref_info.tile_height;
    }
    chunk_height = std::min(chunk_height, height);
    chunk_height = std::max(chunk_height, ref_info.tile_height);

    int num_chunks = (height + chunk_height - 1) / chunk_height;

    stack_ensure_pool(N, chunk_height, width);

    
    std::vector<GDALDataset*> datasets(N, nullptr);
    for (int s = 0; s < N; ++s) {
        datasets[s] = static_cast<GDALDataset*>(GDALOpen(scene_files[s].c_str(), GA_ReadOnly));
        if (!datasets[s]) throw std::runtime_error("Cannot open scene: " + scene_files[s]);
    }

    
    const TemporalParams& tp = ctx.temporal_params;
    int t0_i  = tp.t0_idx;
    int t1_i  = (tp.t1_idx < 0) ? (N - 1) : tp.t1_idx;
    int op_id = static_cast<int>(tp.op);

    
    float* d_time_values = nullptr;
    std::vector<float> time_vals = tp.time_values;
    float denominator = tp.denominator;

    if (tp.op == TemporalOp::TREND) {
        if (time_vals.empty()) {
            time_vals.resize(N);
            std::iota(time_vals.begin(), time_vals.end(), 0.f);
        }
        if (denominator == 1.f) {
            float n = (float)N;
            float sum_t = 0, sum_t2 = 0;
            for (float t : time_vals) { sum_t += t; sum_t2 += t * t; }
            denominator = n * sum_t2 - sum_t * sum_t;
            if (fabsf(denominator) < 1e-10f) denominator = 1.f;
        }
        CUDA_CHECK(cudaMalloc(&d_time_values, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_time_values, time_vals.data(),
                              N * sizeof(float), cudaMemcpyHostToDevice));
    }

    
    
    
    

    std::vector<float*> h_scene_a(N, nullptr);
    size_t per_scene_bytes = (size_t)width * chunk_height * sizeof(float);
    for (int s = 0; s < N; ++s) {
        CUDA_CHECK(cudaHostAlloc(&h_scene_a[s], per_scene_bytes,
                                 cudaHostAllocDefault));
        memset(h_scene_a[s], 0, per_scene_bytes);
    }

    
    
    

    
    float* h_out_a = g_stack_buf.h_output;
    float* d_out_a = g_stack_buf.d_output;  
    
    float* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, (size_t)width * chunk_height * sizeof(float)));

    double progress_state = -1.0;
    if (verbose) GDALTermProgress(0.0, nullptr, &progress_state);

    
    {
        int cur_h = std::min(chunk_height, height);
        for (int s = 0; s < N; ++s) {
            GDALRasterBand* bd = datasets[s]->GetRasterBand(1);
            bd->RasterIO(GF_Read, 0, 0, width, cur_h,
                         h_scene_a[s], width, cur_h, GDT_Float32, 0, 0);
        }
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int y0    = chunk * chunk_height;
        int cur_h = std::min(chunk_height, height - y0);
        size_t cur_pixels = (size_t)width * cur_h;

        
        
        cudaStream_t stream_a = g_stack_buf.cuda_stream;

        for (int s = 0; s < N; ++s) {
            float* dst_s = g_stack_buf.d_stack_output
                         + static_cast<size_t>(s) * (width * chunk_height);
            CUDA_CHECK(cudaMemcpyAsync(dst_s, h_scene_a[s],
                                       cur_pixels * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_a));
        }

        
        if (chunk + 1 < num_chunks) {
            int ny0    = (chunk + 1) * chunk_height;
            int ncur_h = std::min(chunk_height, height - ny0);
            for (int s = 0; s < N; ++s) {
                GDALRasterBand* bd = datasets[s]->GetRasterBand(1);
                bd->RasterIO(GF_Read, 0, ny0, width, ncur_h,
                             g_scene_bufs_b[s], width, ncur_h, GDT_Float32, 0, 0);
            }
        }

        
        CUDA_CHECK(cudaStreamSynchronize(stream_a));

        launch_temporal_kernel(
            const_cast<const float**>(
                reinterpret_cast<float**>(g_stack_buf.d_stack_band_ptrs)),
            d_result, cur_pixels, N, op_id, t0_i, t1_i,
            d_time_values, denominator, stream_a);

        
        if (chunk + 1 < num_chunks) {
            int ncur_h = std::min(chunk_height, height - (chunk+1)*chunk_height);
            size_t nbytes = (size_t)width * ncur_h * sizeof(float);
            for (int s = 0; s < N; ++s) {
                memcpy(h_scene_a[s], g_scene_bufs_b[s], nbytes);
            }
        }

        
        CUDA_CHECK(cudaStreamSynchronize(stream_a));

        CUDA_CHECK(cudaMemcpy(h_out_a, d_result,
                              cur_pixels * sizeof(float), cudaMemcpyDeviceToHost));

        
        if (ctx.output_band) {
#pragma omp critical(gdal_write)
            ctx.output_band->RasterIO(GF_Write, 0, y0, width, cur_h,
                                       h_out_a, width, cur_h, GDT_Float32, 0, 0);
        }
        if (ctx.queue_callback)  { ctx.queue_callback(width, cur_h, h_out_a, y0); }
        if (ctx.result_callback) { ctx.result_callback(width, cur_h, h_out_a, y0); }

        if (verbose) {
            GDALTermProgress((double)(chunk+1)/num_chunks, nullptr, &progress_state);
        }
    }

    cudaFree(d_result);
    if (d_time_values) cudaFree(d_time_values);
    
    for (int s = 0; s < N; ++s) {
        if (h_scene_a[s]) cudaFreeHost(h_scene_a[s]);
    }
    for (auto* ds : datasets) { if (ds) GDALClose(ds); }
    if (verbose) GDALTermProgress(1.0, nullptr, &progress_state);
}
