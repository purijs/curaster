#include "engine.h"
#include "engine_focal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <omp.h>
#include "gdal_priv.h"
#include "cpl_progress.h"

#include "../../include/cuda_utils.h"
#include "../../include/raster_core.h"
#include "../../include/thread_buffers.h"
#include "../../include/pipeline.h"

#include "../tiff/tiff_metadata.h"



struct HaloPoolKey {
    size_t halo_bytes;
    int    out_bands;
    int    num_threads;
    size_t max_chunk_pixels;
    int    radius;
    size_t vram_snapshot_mb;
    bool operator==(const HaloPoolKey& o) const noexcept {
        return halo_bytes == o.halo_bytes && out_bands == o.out_bands
            && num_threads == o.num_threads && max_chunk_pixels == o.max_chunk_pixels
            && radius == o.radius;
        
    }
};

static std::vector<ThreadBufs> g_halo_pool;
static HaloPoolKey              g_halo_key{0,0,0,0,0,0};



static std::vector<float*> g_halo_ping_b;
static std::vector<float*> g_halo_ping_b_d;

static std::vector<cudaStream_t> g_stream_b;

static std::vector<ThreadBufs>& get_persistent_halo_pool(
    size_t halo_bytes, int num_out_bands, int num_threads, size_t max_chunk_pixels,
    int radius, size_t band_bytes, int num_src_bands, size_t current_free_vram_mb)
{
    HaloPoolKey needed{halo_bytes, num_out_bands, num_threads, max_chunk_pixels,
                       radius, current_free_vram_mb};

    
    
    
    bool vram_dropped = (current_free_vram_mb + 512 < g_halo_key.vram_snapshot_mb);

    if (needed == g_halo_key && !g_halo_pool.empty() && !vram_dropped) {
        return g_halo_pool;
    }

    
    for (auto& buf : g_halo_pool) { buf.free_all(); }
    g_halo_pool.clear();
    for (float* p : g_halo_ping_b) { if (p) cudaFreeHost(p); }
    g_halo_ping_b.clear();
    g_halo_ping_b_d.clear();
    for (auto s : g_stream_b) { cudaStreamDestroy(s); }
    g_stream_b.clear();

    g_halo_pool.resize(num_threads);
    g_halo_ping_b.resize(num_threads, nullptr);
    g_halo_ping_b_d.resize(num_threads, nullptr);
    g_stream_b.resize(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        g_halo_pool[i].alloc(band_bytes, num_src_bands, 0);
        g_halo_pool[i].alloc_halo(halo_bytes, num_out_bands, max_chunk_pixels);

        
        CUDA_CHECK(cudaHostAlloc(&g_halo_ping_b[i], halo_bytes,
                                 cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(g_halo_ping_b[i], 0, halo_bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&g_halo_ping_b_d[i]), g_halo_ping_b[i], 0));

        
        CUDA_CHECK(cudaStreamCreate(&g_stream_b[i]));
    }
    needed.vram_snapshot_mb = current_free_vram_mb;
    g_halo_key = needed;
    return g_halo_pool;
}

void release_halo_pool() {
    for (auto& buf : g_halo_pool) { buf.free_all(); }
    g_halo_pool.clear();
    for (float* p : g_halo_ping_b) { if (p) cudaFreeHost(p); }
    g_halo_ping_b.clear();
    g_halo_ping_b_d.clear();
    for (auto s : g_stream_b) { cudaStreamDestroy(s); }
    g_stream_b.clear();
    g_halo_key = HaloPoolKey{0,0,0,0,0,0};
}





static void read_halo_chunk(
    GDALRasterBand* band,
    int             chunk_y0,
    int             cur_h,
    int             height,
    int             width,
    int             radius,
    float*          halo_buf)    
{
    memset(halo_buf, 0, (size_t)width * (cur_h + 2 * radius) * sizeof(float));

    int halo_y0 = std::max(0, chunk_y0 - radius);
    int halo_y1 = std::min(height, chunk_y0 + cur_h + radius);
    int actual_h = halo_y1 - halo_y0;
    if (actual_h <= 0) return;

    
    int clamp_top  = chunk_y0 - halo_y0;   
    int buf_row_start = radius - clamp_top; 
    float* write_ptr = halo_buf + (size_t)buf_row_start * width;

    band->RasterIO(GF_Read, 0, halo_y0, width, actual_h,
                   write_ptr, width, actual_h, GDT_Float32, 0, 0);

    
    if (clamp_top > 0 && buf_row_start > 0) {
        float* row0 = halo_buf + (size_t)buf_row_start * width;
        for (int r = 0; r < buf_row_start; ++r) {
            memcpy(halo_buf + (size_t)r * width, row0, width * sizeof(float));
        }
    }
    
    int written_end = buf_row_start + actual_h;        
    int total_rows  = cur_h + 2 * radius;
    if (written_end < total_rows) {
        float* last_row = halo_buf + (size_t)(written_end - 1) * width;
        for (int r = written_end; r < total_rows; ++r) {
            memcpy(halo_buf + (size_t)r * width, last_row, width * sizeof(float));
        }
    }
}



void run_engine_focal(const std::string& input_file, PipelineCtx& ctx, bool verbose) {
    GDALAllRegister();
    
    release_warp_pool();
    release_stack_pool();
    release_zonal_pool();
    g_pinned_arena.release();
    init_ram_budget();

    FileInfo src_info = get_file_info(input_file);
    const int width  = src_info.width;
    const int height = src_info.height;

    
    int radius = 1;
    if (ctx.has_focal)   { radius = ctx.focal_params.radius; }
    if (ctx.has_terrain) { radius = 1; }
    if (ctx.has_texture) { radius = ctx.glcm_params.window / 2; }

    int num_out_bands = ctx.focal_num_output_bands;
    if (num_out_bands <= 0) num_out_bands = 1;

    
    
    
    

    const int tile_h = src_info.tile_height;
    int num_src_bands = 1;  

    size_t free_vram = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
    size_t vram_budget      = static_cast<size_t>(free_vram * 0.80);
    size_t current_free_mb  = free_vram / (1024 * 1024);

    
    int num_threads = std::max(1, omp_get_max_threads());
    g_num_threads   = num_threads;

    
    int chunk_height = tile_h;
    {
        size_t bytes_per_pixel_halo = (size_t)num_src_bands * sizeof(float);
        size_t halo_extra_bytes     = 2 * radius * width * num_src_bands * sizeof(float);

        while (chunk_height + tile_h <= height) {
            size_t per_thread_pinned =
                2 * (size_t)width * (chunk_height + 2 * radius) * bytes_per_pixel_halo
              + (size_t)width * chunk_height * (num_src_bands + 1) * sizeof(float);
            if (per_thread_pinned * num_threads > (size_t)(g_pinned_budget * 0.80)) break;

            size_t per_thread_vram = 2 * halo_extra_bytes
                                   + (size_t)width * chunk_height * num_out_bands * sizeof(float);
            if (per_thread_vram * num_threads > vram_budget) break;
            chunk_height += tile_h;
        }

        chunk_height = std::min(chunk_height, height);
        (void)halo_extra_bytes;
    }

    const int num_chunks = (height + chunk_height - 1) / chunk_height;
    size_t halo_h_full   = chunk_height + 2 * radius;
    size_t halo_bytes    = (size_t)width * halo_h_full * sizeof(float);
    size_t max_chunk_pixels = (size_t)width * chunk_height;
    size_t band_bytes       = max_chunk_pixels * sizeof(float);

    
    {
        size_t per_t_vram = 2 * halo_bytes + max_chunk_pixels * num_out_bands * sizeof(float);
        if (per_t_vram > 0) {
            int max_t = (int)std::max(size_t(1), vram_budget / per_t_vram);
            num_threads = std::min(num_threads, max_t);
        }
    }

    auto& pool = get_persistent_halo_pool(halo_bytes, num_out_bands, num_threads,
                                           max_chunk_pixels, radius,
                                           band_bytes, num_src_bands, current_free_mb);

    
    std::vector<GDALDataset*> datasets(num_threads, nullptr);
    std::vector<int>          band_slots = ctx.band_map;
    if (band_slots.empty()) band_slots.push_back(0);

    for (int t = 0; t < num_threads; ++t) {
        datasets[t] = static_cast<GDALDataset*>(GDALOpen(input_file.c_str(), GA_ReadOnly));
        if (!datasets[t]) throw std::runtime_error("Cannot open: " + input_file);
    }

    
    float glcm_val_min = ctx.glcm_params.value_min;
    float glcm_val_max = ctx.glcm_params.value_max;
    if (ctx.has_texture && (glcm_val_min == 0.f && glcm_val_max == 0.f)) {
        GDALRasterBand* scan_band = datasets[0]->GetRasterBand(band_slots[0] + 1);
        int has_min_flag = 0, has_max_flag = 0;
        glcm_val_min = static_cast<float>(scan_band->GetMinimum(&has_min_flag));
        glcm_val_max = static_cast<float>(scan_band->GetMaximum(&has_max_flag));
        if (!has_min_flag || !has_max_flag) {
            double mn, mx;
            scan_band->ComputeStatistics(FALSE, &mn, &mx,
                                                nullptr, nullptr, nullptr, nullptr);
            glcm_val_min = static_cast<float>(mn);
            glcm_val_max = static_cast<float>(mx);
        }
    }

    
    int terrain_unit_mode = 0;
    if (ctx.has_terrain) {
        const std::string& u = ctx.terrain_params.unit;
        if      (u == "radians") terrain_unit_mode = 1;
        else if (u == "percent") terrain_unit_mode = 2;
    }

    std::atomic<int> completed_chunks{0};
    double           progress_state = -1.0;
    if (verbose) GDALTermProgress(0.0, nullptr, &progress_state);

    
    
    
    
    
    

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int tid    = omp_get_thread_num();
        int y0     = chunk * chunk_height;
        int cur_h  = std::min(chunk_height, height - y0);
        ThreadBufs& buf = pool[tid];

        
        bool use_b  = (chunk & 1) != 0;
        float* halo_host_cur = use_b ? g_halo_ping_b[tid]  : buf.h_halo_master;
        float* halo_dev_cur  = use_b ? g_halo_ping_b_d[tid] : buf.d_halo_device;
        cudaStream_t stream_cur = use_b ? g_stream_b[tid] : buf.cuda_stream;

        
        GDALRasterBand* bd = datasets[tid]->GetRasterBand(band_slots[0] + 1);
        read_halo_chunk(bd, y0, cur_h, height, width, radius, halo_host_cur);

        
        
        
        cudaStream_t stream_prev = use_b ? buf.cuda_stream : g_stream_b[tid];
        CUDA_CHECK(cudaStreamSynchronize(stream_prev));

        
        int halo_h_kernel = cur_h + 2 * radius;

        if (ctx.has_focal) {
            int stat_id      = static_cast<int>(ctx.focal_params.stat);
            int shape_circle = (ctx.focal_params.shape == FocalShape::CIRCLE) ? 1 : 0;
            launch_focal_kernel(
                halo_dev_cur, buf.d_neighborhood_output,
                width, halo_h_kernel, width, cur_h,
                ctx.focal_params.radius, stat_id, shape_circle, stream_cur);

        } else if (ctx.has_terrain) {
            float sun_az  = static_cast<float>(ctx.terrain_params.sun_azimuth  * M_PI / 180.0);
            float sun_alt = static_cast<float>(ctx.terrain_params.sun_altitude * M_PI / 180.0);
            bool  zev     = (ctx.terrain_params.method == "zevenbergen");
            launch_terrain_kernel(
                halo_dev_cur, buf.d_neighborhood_output,
                width, halo_h_kernel, width, cur_h,
                ctx.terrain_params.features_mask,
                ctx.terrain_params.cell_size_x, ctx.terrain_params.cell_size_y,
                sun_az, sun_alt, zev, terrain_unit_mode,
                max_chunk_pixels, stream_cur);

        } else if (ctx.has_texture) {
            
            static const int dx4[] = {1, 1, 0, -1};
            static const int dy4[] = {0, 1, 1,  1};
            int num_dirs = ctx.glcm_params.avg_directions ? 4 : 1;

            
            CUDA_CHECK(cudaMemsetAsync(buf.d_neighborhood_output, 0,
                max_chunk_pixels * num_out_bands * sizeof(float), stream_cur));

            for (int d = 0; d < num_dirs; ++d) {
                launch_glcm_kernel(
                    halo_dev_cur, buf.d_neighborhood_output,
                    width, halo_h_kernel, width, cur_h,
                    ctx.glcm_params.window, ctx.glcm_params.levels,
                    glcm_val_min, glcm_val_max,
                    dx4[d], dy4[d], 18, max_chunk_pixels,
                    ctx.glcm_params.log_scale, stream_cur);
            }

            
            if (num_dirs > 1) {
                launch_glcm_avg_divide(
                    buf.d_neighborhood_output, (int)max_chunk_pixels,
                    18, num_dirs, stream_cur);
            }
        }

        
        
        
        size_t cur_pixels = (size_t)width * cur_h;

        if (num_out_bands == 1) {
            CUDA_CHECK(cudaMemcpyAsync(buf.d_output, buf.d_neighborhood_output,
                                       cur_pixels * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream_cur));
            CUDA_CHECK(cudaStreamSynchronize(stream_cur));

            if (ctx.output_band) {
#pragma omp critical(gdal_write)
                ctx.output_band->RasterIO(GF_Write, 0, y0, width, cur_h,
                                           buf.h_output, width, cur_h,
                                           GDT_Float32, 0, 0);
            }
            if (ctx.queue_callback)  { ctx.queue_callback(width, cur_h, buf.h_output, y0); }
            if (ctx.result_callback) { ctx.result_callback(width, cur_h, buf.h_output, y0); }

        } else {
            CUDA_CHECK(cudaStreamSynchronize(stream_cur));
            std::vector<float> mb_host(max_chunk_pixels * num_out_bands);
            CUDA_CHECK(cudaMemcpy(mb_host.data(), buf.d_neighborhood_output,
                                  max_chunk_pixels * num_out_bands * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            if (ctx.output_dataset) {
                GDALDataset* out_ds = static_cast<GDALDataset*>(ctx.output_dataset);
#pragma omp critical(gdal_write)
                for (int b = 0; b < num_out_bands; ++b) {
                    out_ds->GetRasterBand(b + 1)->RasterIO(
                        GF_Write, 0, y0, width, cur_h,
                        mb_host.data() + (size_t)b * max_chunk_pixels, width, cur_h,
                        GDT_Float32, 0, 0);
                }
            }

            if (ctx.result_callback || ctx.queue_callback) {
                std::vector<float> compact(cur_pixels * num_out_bands);
                for (int b = 0; b < num_out_bands; ++b) {
                    memcpy(compact.data() + (size_t)b * cur_pixels,
                           mb_host.data() + (size_t)b * max_chunk_pixels,
                           cur_pixels * sizeof(float));
                }
                if (ctx.queue_callback)  { ctx.queue_callback(width, cur_h, compact.data(), y0); }
                if (ctx.result_callback) { ctx.result_callback(width, cur_h, compact.data(), y0); }
            }
        }

        if (verbose) {
            int done = ++completed_chunks;
#pragma omp critical
            GDALTermProgress((double)done / num_chunks, nullptr, &progress_state);
        }
    }

    
    for (int t = 0; t < num_threads; ++t) {
        cudaStreamSynchronize(pool[t].cuda_stream);
        cudaStreamSynchronize(g_stream_b[t]);
    }

    for (auto* ds : datasets) { if (ds) GDALClose(ds); }
    if (verbose) GDALTermProgress(1.0, nullptr, &progress_state);
}
