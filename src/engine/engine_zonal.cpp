#include "engine.h"
#include "engine_zonal.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <float.h>
#include <stdexcept>

#include "gdal_priv.h"
#include "cpl_progress.h"
#include <omp.h>

#include "../../include/cuda_utils.h"
#include "../../include/raster_core.h"
#include "../../include/thread_buffers.h"
#include "../../include/pipeline.h"

#include "../tiff/tiff_metadata.h"
#include "../zonal/zonal.h"





struct ZonalPoolKey {
    int    num_zones;
    int    chunk_height;
    int    width;
    bool operator==(const ZonalPoolKey& o) const noexcept {
        return num_zones == o.num_zones && chunk_height == o.chunk_height && width == o.width;
    }
};

static ThreadBufs        g_zonal_buf;
static ZonalPoolKey      g_zonal_key{0, 0, 0};
static bool              g_zonal_init = false;


static uint16_t* g_zonal_label_b     = nullptr;
static uint16_t* g_zonal_label_b_dev = nullptr;
static float*    g_zonal_val_b       = nullptr;
static float*    g_zonal_val_b_dev   = nullptr;
static cudaStream_t g_zonal_stream_b{};

static void zonal_ensure_pool(int num_zones, int chunk_height, int width) {
    ZonalPoolKey needed{num_zones, chunk_height, width};
    if (g_zonal_init && needed == g_zonal_key) return;

    if (g_zonal_init) {
        g_zonal_buf.free_zonal();
        g_zonal_buf.free_all();
        if (g_zonal_label_b)  { cudaFreeHost(g_zonal_label_b); g_zonal_label_b = nullptr; }
        if (g_zonal_val_b)    { cudaFreeHost(g_zonal_val_b);   g_zonal_val_b   = nullptr; }
        cudaStreamDestroy(g_zonal_stream_b);
    }

    
    size_t pixel_bytes  = (size_t)width * chunk_height * sizeof(float);
    size_t label_bytes  = (size_t)width * chunk_height * sizeof(uint16_t);

    
    g_zonal_buf.alloc(pixel_bytes, 1, 0); 
    g_zonal_buf.alloc_zonal(num_zones, chunk_height, width);

    
    CUDA_CHECK(cudaMemset(g_zonal_buf.d_zone_count,   0, (num_zones+1)*sizeof(int)));
    CUDA_CHECK(cudaMemset(g_zonal_buf.d_zone_sum,     0, (num_zones+1)*sizeof(float)));
    CUDA_CHECK(cudaMemset(g_zonal_buf.d_zone_sum_sq,  0, (num_zones+1)*sizeof(float)));
    
    std::vector<float> init_min(num_zones+1,  FLT_MAX);
    std::vector<float> init_max(num_zones+1, -FLT_MAX);
    CUDA_CHECK(cudaMemcpy(g_zonal_buf.d_zone_min_buf, init_min.data(),
                          (num_zones+1)*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_zonal_buf.d_zone_max_buf, init_max.data(),
                          (num_zones+1)*sizeof(float), cudaMemcpyHostToDevice));

    
    CUDA_CHECK(cudaHostAlloc(&g_zonal_label_b, label_bytes, cudaHostAllocMapped));
    memset(g_zonal_label_b, 0, label_bytes);
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&g_zonal_label_b_dev), g_zonal_label_b, 0));

    
    CUDA_CHECK(cudaHostAlloc(&g_zonal_val_b, pixel_bytes, cudaHostAllocMapped));
    memset(g_zonal_val_b, 0, pixel_bytes);
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&g_zonal_val_b_dev), g_zonal_val_b, 0));

    CUDA_CHECK(cudaStreamCreate(&g_zonal_stream_b));
    g_zonal_key   = needed;
    g_zonal_init  = true;
}

void release_zonal_pool() {
    if (!g_zonal_init) return;
    g_zonal_buf.free_zonal();
    g_zonal_buf.free_all();
    if (g_zonal_label_b) { cudaFreeHost(g_zonal_label_b); g_zonal_label_b = nullptr; }
    if (g_zonal_val_b)   { cudaFreeHost(g_zonal_val_b);   g_zonal_val_b   = nullptr; }
    if (g_zonal_stream_b) { cudaStreamDestroy(g_zonal_stream_b); g_zonal_stream_b = {}; }
    g_zonal_key  = ZonalPoolKey{0, 0, 0};
    g_zonal_init = false;
}



void run_engine_zonal(const std::string& input_file, PipelineCtx& ctx, bool verbose) {
    GDALAllRegister();
    release_warp_pool();
    release_halo_pool();
    release_stack_pool();
    g_pinned_arena.release();
    init_ram_budget();

    FileInfo src_info = get_file_info(input_file);
    int width  = src_info.width;
    int height = src_info.height;

    int chunk_height = src_info.tile_height;
    
    {
        size_t bytes_per_row = (size_t)width * (sizeof(float) + sizeof(uint16_t));
        size_t max_rows = g_pinned_budget / bytes_per_row / 4; 
        while (chunk_height + src_info.tile_height <= (int)max_rows &&
               chunk_height + src_info.tile_height <= height) {
            chunk_height += src_info.tile_height;
        }
    }
    chunk_height = std::min(chunk_height, height);

    int num_chunks = (height + chunk_height - 1) / chunk_height;

    auto prebuilt_zones = build_prebuilt_zones(ctx.zonal_params.geojson_str, src_info);
    int  num_zones = static_cast<int>(prebuilt_zones.size());
    if (num_zones == 0) return;

    int band_idx = std::max(1, ctx.zonal_params.band) - 1;
    band_idx = std::min(band_idx, src_info.samples_per_pixel - 1);

    zonal_ensure_pool(num_zones, chunk_height, width);

    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(input_file.c_str(), GA_ReadOnly));
    if (!ds) throw std::runtime_error("run_engine_zonal: cannot open " + input_file);
    GDALRasterBand* band = ds->GetRasterBand(band_idx + 1);

    
    
    float*    val_bufs[2]   = { g_zonal_buf.h_bands[0],  g_zonal_val_b };
    float*    val_devs[2]   = { g_zonal_buf.d_bands[0],  g_zonal_val_b_dev };
    uint16_t* lbl_bufs[2]   = { g_zonal_buf.h_zone_labels, g_zonal_label_b };
    uint16_t* lbl_devs[2]   = { g_zonal_buf.d_zone_labels, g_zonal_label_b_dev };
    cudaStream_t streams[2] = { g_zonal_buf.cuda_stream, g_zonal_stream_b };

    double progress_state = -1.0;
    if (verbose) GDALTermProgress(0.0, nullptr, &progress_state);

    
    int slot = 0;
    {
        int y0    = 0;
        int cur_h = std::min(chunk_height, height);
        band->RasterIO(GF_Read, 0, y0, width, cur_h,
                       val_bufs[slot], width, cur_h, GDT_Float32, 0, 0);
        rasterize_zones_prebuilt(prebuilt_zones, src_info,
                                  y0, cur_h, lbl_bufs[slot]);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int y0    = chunk * chunk_height;
        int cur_h = std::min(chunk_height, height - y0);
        size_t cur_pixels = (size_t)width * cur_h;

        
        int cur_slot  = slot;
        int next_slot = 1 - slot;

        
        launch_zonal_reduction(
            val_devs[cur_slot], lbl_devs[cur_slot], cur_pixels,
            g_zonal_buf.d_zone_count, g_zonal_buf.d_zone_sum,
            g_zonal_buf.d_zone_sum_sq, g_zonal_buf.d_zone_min_buf,
            g_zonal_buf.d_zone_max_buf, num_zones, streams[cur_slot]);

        if (chunk + 1 < num_chunks) {
            int ny0    = (chunk + 1) * chunk_height;
            int ncur_h = std::min(chunk_height, height - ny0);
            band->RasterIO(GF_Read, 0, ny0, width, ncur_h,
                           val_bufs[next_slot], width, ncur_h, GDT_Float32, 0, 0);
            rasterize_zones_prebuilt(prebuilt_zones, src_info,
                                      ny0, ncur_h, lbl_bufs[next_slot]);
        }

        
        CUDA_CHECK(cudaStreamSynchronize(streams[cur_slot]));

        if (verbose) {
            double frac = (double)(chunk + 1) / num_chunks;
            GDALTermProgress(frac, nullptr, &progress_state);
        }

        slot = next_slot;
    }

    
    int n = num_zones + 1;
    std::vector<int>   h_count(n);
    std::vector<float> h_sum(n), h_sum_sq(n), h_min(n), h_max(n);
    CUDA_CHECK(cudaMemcpy(h_count.data(),  g_zonal_buf.d_zone_count,   n*sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum.data(),    g_zonal_buf.d_zone_sum,     n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_sq.data(), g_zonal_buf.d_zone_sum_sq,  n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min.data(),    g_zonal_buf.d_zone_min_buf, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(),    g_zonal_buf.d_zone_max_buf, n*sizeof(float), cudaMemcpyDeviceToHost));

    ctx.zonal_results = aggregate_zonal_results(
        h_count.data(), h_sum.data(), h_sum_sq.data(),
        h_min.data(), h_max.data(), num_zones, ctx.zonal_params.stats);

    if (verbose) GDALTermProgress(1.0, nullptr, &progress_state);
    GDALClose(ds);
}
