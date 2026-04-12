/**
 * @file engine.cpp
 * @brief Direct-read (non-warp) pipeline engine.
 *
 * Reads source tiles / strips into pinned host memory per OMP thread, launches
 * the GPU algebra and optional mask kernels, then delivers results via the
 * callbacks in PipelineCtx.
 */
#include "engine.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gdal_priv.h"
#include "cpl_progress.h"

#include "../../include/cuda_utils.h"
#include "../../include/thread_buffers.h"
#include "../../include/raster_core.h"

#include "../s3/s3_auth.h"
#include "../s3/s3_fetch.h"
#include "../tiff/tiff_metadata.h"
#include "../tile_io/tile_io.h"

// ─── Globals ──────────────────────────────────────────────────────────────────

int    g_num_threads  = std::max(1, std::min(omp_get_max_threads(), 11));
size_t g_pinned_budget = 0;

// ─── RAM budget ───────────────────────────────────────────────────────────────

size_t get_available_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullAvailPhys);
#else
    size_t available_pages = static_cast<size_t>(sysconf(_SC_AVPHYS_PAGES));
    size_t page_size       = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
    return available_pages * page_size;
#endif
}

void init_ram_budget() {
    if (g_pinned_budget != 0) { return; }
    g_pinned_budget = static_cast<size_t>(get_available_ram() * 0.55);
}

// ─── Local GDAL dataset container ────────────────────────────────────────────

/// Owns multiple GDALDataset handles (one per thread) and closes them on destroy.
struct DatasetPool {
    std::vector<GDALDataset*>              datasets;
    std::vector<std::vector<GDALRasterBand*>> bands;

    DatasetPool(int num_threads, int num_bands)
        : datasets(num_threads, nullptr)
        , bands(num_threads, std::vector<GDALRasterBand*>(num_bands, nullptr)) {}

    ~DatasetPool() {
        for (auto* ds : datasets) { if (ds) { GDALClose(ds); } }
    }
};

// ─── run_engine_ex ───────────────────────────────────────────────────────────

void run_engine_ex(const std::string& input_file, PipelineCtx& ctx, bool verbose) {
    // Dispatch to the warp engine if reprojection is requested.
    if (ctx.has_reproject) {
        run_engine_reproject(input_file, ctx, verbose);
        return;
    }

    GDALAllRegister();
    init_ram_budget();

    FileInfo file_info          = get_file_info(input_file);
    const bool is_pixel_interleaved = (file_info.interleave == "PIXEL");
    const bool is_s3            = is_s3_path(input_file);
    S3Loc s3_location;
    if (is_s3) { s3_location = parse_s3_path(input_file); }

    const std::vector<int>& band_slots = ctx.band_map;
    const int num_bands = static_cast<int>(band_slots.size());

    // ── Chunk size calculation ─────────────────────────────────────────────
    // Choose the largest chunk height that fits within the pinned RAM budget,
    // aligned to tile block boundaries.
    size_t bytes_per_row      = static_cast<size_t>(file_info.width) * (num_bands + 1) * sizeof(float);
    size_t max_rows_in_budget = static_cast<size_t>(
        g_pinned_budget / (static_cast<size_t>(g_num_threads) * bytes_per_row));
    int chunk_height = file_info.tile_height;
    while (chunk_height + file_info.tile_height <= static_cast<int>(max_rows_in_budget)) {
        chunk_height += file_info.tile_height;
    }
    chunk_height = std::min(chunk_height, file_info.height);

    int    num_chunks  = (file_info.height + chunk_height - 1) / chunk_height;
    size_t band_bytes  = static_cast<size_t>(file_info.width) * chunk_height * sizeof(float);

    // ── Allocate per-thread buffers ────────────────────────────────────────
    std::vector<ThreadBufs> thread_pool(g_num_threads);
    for (int thread_idx = 0; thread_idx < g_num_threads; ++thread_idx) {
        thread_pool[thread_idx].alloc(band_bytes, num_bands,
                                      ctx.has_clip_mask ? chunk_height : 0);
    }

    // ── Open GDAL datasets (non-S3 path only) ─────────────────────────────
    DatasetPool dataset_pool(g_num_threads, num_bands);
    if (!is_s3) {
        for (int thread_idx = 0; thread_idx < g_num_threads; ++thread_idx) {
            dataset_pool.datasets[thread_idx] = static_cast<GDALDataset*>(
                GDALOpen(input_file.c_str(), GA_ReadOnly));
            if (!dataset_pool.datasets[thread_idx]) {
                throw std::runtime_error("Open failed: " + input_file);
            }
            for (int b = 0; b < num_bands; ++b) {
                dataset_pool.bands[thread_idx][b] =
                    dataset_pool.datasets[thread_idx]->GetRasterBand(band_slots[b] + 1);
            }
        }
    }

    // ── Upload compiled algebra program to GPU ─────────────────────────────
    Instruction* d_instructions = nullptr;
    if (!ctx.instructions.empty()) {
        CUDA_CHECK(cudaMalloc(&d_instructions,
                              ctx.instructions.size() * sizeof(Instruction)));
        CUDA_CHECK(cudaMemcpy(d_instructions,
                              ctx.instructions.data(),
                              ctx.instructions.size() * sizeof(Instruction),
                              cudaMemcpyHostToDevice));
    }

    // ── Tile layout helpers ────────────────────────────────────────────────
    size_t tiles_across = ((size_t)file_info.width  + file_info.tile_width  - 1) / file_info.tile_width;
    size_t tiles_down   = ((size_t)file_info.height + file_info.tile_height - 1) / file_info.tile_height;

    // 1-indexed band numbers for GDAL RasterIO panBandMap.
    std::vector<int> gdal_band_nums(num_bands);
    for (int b = 0; b < num_bands; ++b) { gdal_band_nums[b] = band_slots[b] + 1; }

    std::atomic<int> completed_chunks{0};
    double progress_state = -1.0;
    if (verbose) { GDALTermProgress(0.0, nullptr, &progress_state); }

    // ── Parallel chunk processing ──────────────────────────────────────────
    omp_set_num_threads(g_num_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int          thread_id  = omp_get_thread_num();
        int          chunk_y0   = chunk * chunk_height;
        int          cur_height = std::min(chunk_height, file_info.height - chunk_y0);
        size_t       num_pixels = static_cast<size_t>(file_info.width) * cur_height;
        ThreadBufs&  buf        = thread_pool[thread_id];

        // ── Phase 1: Read band data ──────────────────────────────────────
        if (!is_s3) {
            // GDAL path: advise read then call RasterIO.
            dataset_pool.datasets[thread_id]->AdviseRead(
                0, chunk_y0, file_info.width, cur_height,
                file_info.width, cur_height, GDT_Float32,
                num_bands, nullptr, nullptr);

            if (is_pixel_interleaved) {
                // Read all requested bands in one interleaved RasterIO call.
                std::vector<float> interleaved_tmp(
                    static_cast<size_t>(file_info.width) * cur_height * num_bands);

                dataset_pool.datasets[thread_id]->RasterIO(
                    GF_Read, 0, chunk_y0, file_info.width, cur_height,
                    interleaved_tmp.data(), file_info.width, cur_height,
                    GDT_Float32, num_bands, gdal_band_nums.data(),
                    static_cast<int>(sizeof(float) * num_bands),
                    static_cast<int>(sizeof(float) * num_bands * file_info.width),
                    sizeof(float));

                // De-interleave into separate per-band host buffers.
                for (int b = 0; b < num_bands; ++b) {
                    for (size_t px = 0; px < static_cast<size_t>(file_info.width) * cur_height; ++px) {
                        buf.h_bands[b][px] = interleaved_tmp[px * num_bands + b];
                    }
                }
            } else {
                // Planar: read each band independently.
                for (int b = 0; b < num_bands; ++b) {
                    dataset_pool.bands[thread_id][b]->RasterIO(
                        GF_Read, 0, chunk_y0, file_info.width, cur_height,
                        buf.h_bands[b], file_info.width, cur_height,
                        GDT_Float32, 0, 0);
                }
            }
        } else {
            // S3 direct-read path: fetch raw tiles/strips and decompress.
            memset(buf.h_pinned_master, 0, band_bytes * num_bands);
            std::vector<TileFetch> fetch_jobs;

            if (file_info.is_tiled) {
                int tile_row_first = chunk_y0 / file_info.tile_height;
                int tile_row_last  = (chunk_y0 + cur_height - 1) / file_info.tile_height;

                for (int tr = tile_row_first; tr <= tile_row_last; ++tr) {
                    for (int tc = 0; tc < static_cast<int>(tiles_across); ++tc) {
                        if (is_pixel_interleaved) {
                            size_t tile_index = static_cast<size_t>(tr) * tiles_across + tc;
                            if (tile_index < file_info.tile_offsets.size()) {
                                fetch_jobs.push_back({tile_index,
                                    file_info.tile_offsets[tile_index],
                                    file_info.tile_lengths[tile_index], {}, 0});
                            }
                        } else {
                            for (int b = 0; b < num_bands; ++b) {
                                size_t tile_index = static_cast<size_t>(band_slots[b])
                                                  * tiles_across * tiles_down
                                                  + static_cast<size_t>(tr) * tiles_across + tc;
                                if (tile_index < file_info.tile_offsets.size()) {
                                    fetch_jobs.push_back({tile_index,
                                        file_info.tile_offsets[tile_index],
                                        file_info.tile_lengths[tile_index], {}, 0});
                                }
                            }
                        }
                    }
                }

                s3_fetch_tiles(s3_location, file_info.tile_offsets, file_info.tile_lengths,
                               fetch_jobs);

                for (auto& job : fetch_jobs) {
                    if (job.err || job.data.empty()) { continue; }

                    // Use the true samples_per_pixel for pixel-interleaved decompression.
                    int block_spp = is_pixel_interleaved ? file_info.samples_per_pixel : 1;
                    auto float_tile = decompress_tile_to_float(
                        job.data.data(), job.data.size(),
                        file_info.tile_width, file_info.tile_height,
                        block_spp, file_info.compression,
                        file_info.predictor, file_info.data_type);

                    size_t ti = job.tile_index;
                    int tile_row, tile_col, band_plane = -1;
                    if (is_pixel_interleaved) {
                        tile_row = static_cast<int>(ti / tiles_across);
                        tile_col = static_cast<int>(ti % tiles_across);
                    } else {
                        int b    = static_cast<int>(ti / (tiles_across * tiles_down));
                        tile_row = static_cast<int>((ti % (tiles_across * tiles_down)) / tiles_across);
                        tile_col = static_cast<int>(ti % tiles_across);
                        band_plane = b;
                    }

                    extract_tile_bands(
                        float_tile.data(), tile_row, tile_col,
                        chunk_y0, cur_height, file_info.width,
                        file_info.tile_width, file_info.tile_height,
                        block_spp, is_pixel_interleaved,
                        band_slots, buf.h_bands, band_plane);
                }

            } else {
                // Strip layout.
                int strips_per_band = (file_info.height + file_info.rows_per_strip - 1)
                                    / file_info.rows_per_strip;
                int strip_row_first = chunk_y0 / file_info.rows_per_strip;
                int strip_row_last  = (chunk_y0 + cur_height - 1) / file_info.rows_per_strip;

                for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                    int physical_band = band_slots[slot];
                    for (int sr = strip_row_first; sr <= strip_row_last; ++sr) {
                        size_t strip_index = is_pixel_interleaved
                            ? static_cast<size_t>(sr)
                            : static_cast<size_t>(physical_band) * strips_per_band + sr;
                        if (strip_index < file_info.strip_offsets.size()) {
                            fetch_jobs.push_back({strip_index,
                                file_info.strip_offsets[strip_index],
                                file_info.strip_lengths[strip_index], {}, 0});
                        }
                    }
                    if (is_pixel_interleaved) { break; }
                }

                s3_fetch_tiles(s3_location, file_info.strip_offsets, file_info.strip_lengths,
                               fetch_jobs);

                for (auto& job : fetch_jobs) {
                    if (job.err || job.data.empty()) { continue; }

                    int sr = static_cast<int>(job.tile_index);
                    if (!is_pixel_interleaved) { sr = sr % strips_per_band; }

                    int strip_first_row = sr * file_info.rows_per_strip;
                    int strip_num_rows  = std::min(file_info.rows_per_strip,
                                                   file_info.height - strip_first_row);
                    int block_spp = is_pixel_interleaved ? file_info.samples_per_pixel : 1;

                    auto float_strip = decompress_tile_to_float(
                        job.data.data(), job.data.size(),
                        file_info.width, strip_num_rows, block_spp,
                        file_info.compression, file_info.predictor, file_info.data_type);

                    for (size_t slot = 0; slot < band_slots.size(); ++slot) {
                        int band_channel = band_slots[slot];
                        for (int row = std::max(strip_first_row, chunk_y0);
                             row < std::min(strip_first_row + strip_num_rows, chunk_y0 + cur_height);
                             ++row) {
                            int strip_local_row = row - strip_first_row;
                            int chunk_row       = row - chunk_y0;
                            if (is_pixel_interleaved) {
                                for (int x = 0; x < file_info.width; ++x) {
                                    buf.h_bands[slot][
                                        static_cast<size_t>(chunk_row) * file_info.width + x] =
                                        float_strip[static_cast<size_t>(strip_local_row)
                                                    * file_info.width * block_spp
                                                    + x * block_spp + band_channel];
                                }
                            } else {
                                memcpy(
                                    buf.h_bands[slot]
                                        + static_cast<size_t>(chunk_row) * file_info.width,
                                    float_strip.data()
                                        + static_cast<size_t>(strip_local_row) * file_info.width,
                                    file_info.width * sizeof(float));
                            }
                        }
                    }
                }
            }
        }

        // ── Phase 2: Populate clip span table ─────────────────────────────
        if (ctx.has_clip_mask && buf.h_clip_spans) {
            memset(buf.h_clip_spans, 0, cur_height * sizeof(GpuSpanRow));
            for (int row = chunk_y0; row < chunk_y0 + cur_height; ++row) {
                auto it = ctx.clip_spans.find(row);
                if (it != ctx.clip_spans.end() && !it->second.empty()) {
                    buf.h_clip_spans[row - chunk_y0] = it->second[0];
                }
            }
        }

        // ── Phase 3: Launch algebra kernel ────────────────────────────────
        launch_raster_algebra(
            d_instructions,
            static_cast<int>(ctx.instructions.size()),
            const_cast<const float* const*>(buf.d_band_ptrs),
            buf.d_output,
            num_pixels,
            buf.cuda_stream);

        // ── Phase 4: Launch mask kernel (if clip is active) ───────────────
        if (ctx.has_clip_mask && buf.d_clip_spans) {
            launch_apply_mask(buf.d_output, num_pixels,
                              file_info.width, buf.d_clip_spans,
                              buf.cuda_stream);
        }

        // ── Phase 5: Synchronise and deliver results ───────────────────────
        CUDA_CHECK(cudaStreamSynchronize(buf.cuda_stream));

        if (ctx.output_band) {
            #pragma omp critical(gdal_write)
            {
                ctx.output_band->RasterIO(
                    GF_Write, 0, chunk_y0, file_info.width, cur_height,
                    buf.h_output, file_info.width, cur_height,
                    GDT_Float32, 0, 0);
            }
        }
        if (ctx.queue_callback)  { ctx.queue_callback (file_info.width, cur_height, buf.h_output, chunk_y0); }
        if (ctx.result_callback) { ctx.result_callback(file_info.width, cur_height, buf.h_output, chunk_y0); }

        // ── Phase 6: Update progress ──────────────────────────────────────
        if (verbose) {
            int done = ++completed_chunks;
            #pragma omp critical
            GDALTermProgress(static_cast<double>(done) / num_chunks,
                             nullptr, &progress_state);
        }
    }

    if (d_instructions) { cudaFree(d_instructions); }
    for (auto& buf : thread_pool) { buf.free_all(); }
    if (verbose) { GDALTermProgress(1.0, nullptr, &progress_state); }
}
