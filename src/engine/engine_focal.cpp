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

// ─── Persistent halo pool ─────────────────────────────────────────────────────

struct HaloPoolKey {
    size_t halo_bytes;
    int    out_bands;
    int    num_threads;
    size_t max_chunk_pixels;
    int    radius;
    bool operator==(const HaloPoolKey& o) const noexcept {
        return halo_bytes == o.halo_bytes && out_bands == o.out_bands
            && num_threads == o.num_threads && max_chunk_pixels == o.max_chunk_pixels
            && radius == o.radius;
    }
};

static std::vector<ThreadBufs> g_halo_pool;
static HaloPoolKey              g_halo_key{0,0,0,0,0};

// Each ThreadBufs gets TWO halo pinned buffers for ping-pong prefetch.
// Second buffer stored as a parallel array.
static std::vector<float*> g_halo_ping_b;   // "ping b" secondary halo slots (host)
static std::vector<float*> g_halo_ping_b_d; // device ptr for above
// Also two streams per thread for double-buffering
static std::vector<cudaStream_t> g_stream_b;

static std::vector<ThreadBufs>& get_persistent_halo_pool(
    size_t halo_bytes, int num_out_bands, int num_threads, size_t max_chunk_pixels,
    int radius, size_t band_bytes, int num_src_bands)
{
    HaloPoolKey needed{halo_bytes, num_out_bands, num_threads, max_chunk_pixels, radius};
    if (needed == g_halo_key && !g_halo_pool.empty()) {
        return g_halo_pool;
    }

    // Free old pool
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

        // Second halo buffer (ping-b)
        CUDA_CHECK(cudaHostAlloc(&g_halo_ping_b[i], halo_bytes,
                                 cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(g_halo_ping_b[i], 0, halo_bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&g_halo_ping_b_d[i]), g_halo_ping_b[i], 0));

        // Second stream (stream_b — for overlapped processing)
        CUDA_CHECK(cudaStreamCreate(&g_stream_b[i]));
    }
    g_halo_key = needed;
    return g_halo_pool;
}

// ─── Halo read helper ─────────────────────────────────────────────────────────
// Reads halo-expanded rows for a given chunk into the provided buffer.
// Correctly handles image-edge clamping.

static void read_halo_chunk(
    GDALRasterBand* band,
    int             chunk_y0,
    int             cur_h,
    int             height,
    int             width,
    int             radius,
    float*          halo_buf)    // size = width * (cur_h + 2*radius)
{
    memset(halo_buf, 0, (size_t)width * (cur_h + 2 * radius) * sizeof(float));

    int halo_y0 = std::max(0, chunk_y0 - radius);
    int halo_y1 = std::min(height, chunk_y0 + cur_h + radius);
    int actual_h = halo_y1 - halo_y0;
    if (actual_h <= 0) return;

    // The actual data goes at row (radius - (chunk_y0 - halo_y0)) in the buffer
    int clamp_top  = chunk_y0 - halo_y0;   // how many top rows are NOT padded
    int buf_row_start = radius - clamp_top; // where data starts in the halo buffer
    float* write_ptr = halo_buf + (size_t)buf_row_start * width;

    band->RasterIO(GF_Read, 0, halo_y0, width, actual_h,
                   write_ptr, width, actual_h, GDT_Float32, 0, 0);

    // CLAMP border: replicate row 0 into top padding rows
    if (clamp_top > 0 && buf_row_start > 0) {
        float* row0 = halo_buf + (size_t)buf_row_start * width;
        for (int r = 0; r < buf_row_start; ++r) {
            memcpy(halo_buf + (size_t)r * width, row0, width * sizeof(float));
        }
    }
    // CLAMP border: replicate last row into bottom padding rows
    int written_end = buf_row_start + actual_h;        // exclusive row
    int total_rows  = cur_h + 2 * radius;
    if (written_end < total_rows) {
        float* last_row = halo_buf + (size_t)(written_end - 1) * width;
        for (int r = written_end; r < total_rows; ++r) {
            memcpy(halo_buf + (size_t)r * width, last_row, width * sizeof(float));
        }
    }
}

// ─── run_engine_focal ─────────────────────────────────────────────────────────

void run_engine_focal(const std::string& input_file, PipelineCtx& ctx, bool verbose) {
    GDALAllRegister();
    init_ram_budget();

    FileInfo src_info = get_file_info(input_file);
    const int width  = src_info.width;
    const int height = src_info.height;

    // Determine halo radius
    int radius = 1;
    if (ctx.has_focal)   { radius = ctx.focal_params.radius; }
    if (ctx.has_terrain) { radius = 1; }
    if (ctx.has_texture) { radius = ctx.glcm_params.window / 2; }

    int num_out_bands = ctx.focal_num_output_bands;
    if (num_out_bands <= 0) num_out_bands = 1;

    // ── Halo-aware chunk height formula ───────────────────────────────────────
    // bytes_per_thread = width * (chunk_h + 2R) * in_bands * 4  (halo pinned)
    //                  + width * chunk_h * out_bands * 4         (output vram)
    // Find largest chunk_h (multiple of tile_h) satisfying budget.

    const int tile_h = src_info.tile_height;
    int num_src_bands = 1;  // neighborhood ops read one band

    size_t free_vram = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
    size_t vram_budget = static_cast<size_t>(free_vram * 0.80);

    // Find thread count first (use g_num_threads, reduce if tight vram)
    int num_threads = g_num_threads;

    // Compute chunk_height from budget
    int chunk_height = tile_h;
    {
        size_t bytes_per_pixel_halo = (size_t)num_src_bands * sizeof(float);   // source bands
        size_t bytes_per_pixel_out  = (size_t)num_out_bands * sizeof(float);   // output bands
        size_t halo_extra_bytes     = 2 * radius * width * num_src_bands * sizeof(float);
        // pinned budget: width*(ch+2R)*in + width*ch*out per thread
        while (chunk_height + tile_h <= height) {
            size_t per_thread = (size_t)width * (chunk_height + 2 * radius) * bytes_per_pixel_halo
                              + (size_t)width * chunk_height * bytes_per_pixel_out;
            if (per_thread * num_threads > (size_t)(g_pinned_budget * 0.80)) break;
            // VRAM check (both halo ping buffers + output)
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

    // Reduce thread count further based on VRAM
    {
        size_t per_t_vram = 2 * halo_bytes + max_chunk_pixels * num_out_bands * sizeof(float);
        if (per_t_vram > 0) {
            int max_t = (int)std::max(size_t(1), vram_budget / per_t_vram);
            num_threads = std::min(num_threads, max_t);
        }
    }

    auto& pool = get_persistent_halo_pool(halo_bytes, num_out_bands, num_threads,
                                           max_chunk_pixels, radius,
                                           band_bytes, num_src_bands);

    // Open one dataset handle per thread
    std::vector<GDALDataset*> datasets(num_threads, nullptr);
    std::vector<int>          band_slots = ctx.band_map;
    if (band_slots.empty()) band_slots.push_back(0);

    for (int t = 0; t < num_threads; ++t) {
        datasets[t] = static_cast<GDALDataset*>(GDALOpen(input_file.c_str(), GA_ReadOnly));
        if (!datasets[t]) throw std::runtime_error("Cannot open: " + input_file);
    }

    // GLCM auto-range pre-pass
    float glcm_val_min = ctx.glcm_params.value_min;
    float glcm_val_max = ctx.glcm_params.value_max;
    if (ctx.has_texture && (glcm_val_min == 0.f && glcm_val_max == 0.f)) {
        GDALRasterBand* scan_band = datasets[0]->GetRasterBand(band_slots[0] + 1);
        int has_min_flag = 0, has_max_flag = 0;
        glcm_val_min = static_cast<float>(scan_band->GetMinimum(&has_min_flag));
        glcm_val_max = static_cast<float>(scan_band->GetMaximum(&has_max_flag));
        if (!has_min_flag || !has_max_flag) {
            double mn, mx;
            scan_band->ComputeRasterStatistics(FALSE, &mn, &mx,
                                                nullptr, nullptr, nullptr, nullptr);
            glcm_val_min = static_cast<float>(mn);
            glcm_val_max = static_cast<float>(mx);
        }
    }

    // ── Unit mode for terrain ─────────────────────────────────────────────────
    int terrain_unit_mode = 0; // degrees
    if (ctx.has_terrain) {
        const std::string& u = ctx.terrain_params.unit;
        if      (u == "radians") terrain_unit_mode = 1;
        else if (u == "percent") terrain_unit_mode = 2;
    }

    std::atomic<int> completed_chunks{0};
    double           progress_state = -1.0;
    if (verbose) GDALTermProgress(0.0, nullptr, &progress_state);

    // ── Double-buffered prefetch loop ─────────────────────────────────────────
    // Pattern: while GPU runs chunk N on stream_a, CPU reads chunk N+1 into halo_b.
    // Then swap buffers and repeat.

    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int tid    = omp_get_thread_num();
        int y0     = chunk * chunk_height;
        int cur_h  = std::min(chunk_height, height - y0);
        ThreadBufs& buf = pool[tid];

        // Alternate ping-pong buffers: even chunk = use buf.h_halo_master, odd = ping_b
        bool use_b  = (chunk & 1) != 0;
        float* halo_host_cur = use_b ? g_halo_ping_b[tid]  : buf.h_halo_master;
        float* halo_dev_cur  = use_b ? g_halo_ping_b_d[tid] : buf.d_halo_device;
        cudaStream_t stream_cur = use_b ? g_stream_b[tid] : buf.cuda_stream;

        // ── Phase 1: Read halo for THIS chunk ─────────────────────────────────
        GDALRasterBand* bd = datasets[tid]->GetRasterBand(band_slots[0] + 1);
        read_halo_chunk(bd, y0, cur_h, height, width, radius, halo_host_cur);

        // Wait for previous chunk's kernel on this thread's OTHER stream to finish
        // (ensures we don't overwrite the host buffer that the previous GPU kernel
        //  is still consuming via zero-copy)
        cudaStream_t stream_prev = use_b ? buf.cuda_stream : g_stream_b[tid];
        CUDA_CHECK(cudaStreamSynchronize(stream_prev));

        // ── Phase 2: Launch kernel on current stream ───────────────────────────
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
            // GLCM: 4-direction accumulation then divide
            static const int dx4[] = {1, 1, 0, -1};
            static const int dy4[] = {0, 1, 1,  1};
            int num_dirs = ctx.glcm_params.avg_directions ? 4 : 1;

            // Zero output before accumulating across directions
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

            // Divide by num_dirs to get average
            if (num_dirs > 1) {
                launch_glcm_avg_divide(
                    buf.d_neighborhood_output, (int)max_chunk_pixels,
                    18, num_dirs, stream_cur);
            }
        }

        // ── Phase 3: Copy band 0 to d_output for delivery ─────────────────────
        // (for multi-band ops, the full output stays in d_neighborhood_output;
        //  the output_band write below loops over all bands)
        size_t cur_pixels = (size_t)width * cur_h;
        CUDA_CHECK(cudaMemcpyAsync(buf.d_output, buf.d_neighborhood_output,
                                   cur_pixels * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_cur));

        // ── Phase 4: Synchronise and deliver results ───────────────────────────
        CUDA_CHECK(cudaStreamSynchronize(stream_cur));

        // Write all bands to GeoTIFF (if multi-band context)
        if (ctx.output_band) {
            // Single-band write: first output band only
#pragma omp critical(gdal_write)
            ctx.output_band->RasterIO(GF_Write, 0, y0, width, cur_h,
                                       buf.h_output, width, cur_h,
                                       GDT_Float32, 0, 0);
        }
        if (ctx.queue_callback)  { ctx.queue_callback(width, cur_h, buf.h_output, y0); }
        if (ctx.result_callback) { ctx.result_callback(width, cur_h, buf.h_output, y0); }

        if (verbose) {
            int done = ++completed_chunks;
#pragma omp critical
            GDALTermProgress((double)done / num_chunks, nullptr, &progress_state);
        }
    }

    // Final sync: drain any in-flight kernels
    for (int t = 0; t < num_threads; ++t) {
        cudaStreamSynchronize(pool[t].cuda_stream);
        cudaStreamSynchronize(g_stream_b[t]);
    }

    for (auto* ds : datasets) { if (ds) GDALClose(ds); }
    if (verbose) GDALTermProgress(1.0, nullptr, &progress_state);
}
