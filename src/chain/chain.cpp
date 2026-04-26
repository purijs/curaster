#include "chain.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include "gdal_priv.h"
#include "cpl_vsi.h"
#include "cpl_conv.h"

#include "../../include/cuda_utils.h"
#include "../../include/pipeline.h"
#include "../../include/types.h"
#include "../../include/chunk_queue.h"
#include "../../include/vram_cache.h"

#include "../tiff/tiff_metadata.h"
#include "../algebra/algebra_compiler.h"
#include "../clip/clip.h"
#include "../focal/focal.h"
#include "../texture/texture.h"
#include "../engine/engine.h"
#include "../engine/engine_focal.h"
#include "../engine/engine_zonal.h"
#include "../reproject/reproject.h"
#include "../s3/s3_auth.h"
#include "../s3/s3_fetch.h"
#include "../tile_io/tile_io.h"



Chain::Chain(const std::string& input_file)
    : input_file_(input_file) {}

Chain::Chain(const Chain& other)
    : input_file_(other.input_file_)
    , operations_(other.operations_)
    , cache_(other.cache_)
    , user_band_selection_(other.user_band_selection_) {}



// ── VRAM cache loader ────────────────────────────────────────────────────────

/**
 * Load every physical band of @p input_file into VRAM.
 *
 * For each band we allocate:
 *   – A linear cudaMalloc buffer  (algebra engine uses this directly)
 *   – A CUDA 2D array             (warp engine texture backing)
 *   – Two texture objects         (bilinear + nearest, same array)
 *
 * Throws std::runtime_error if:
 *   – decoded data + texture arrays > 80% of currently-free VRAM
 *   – raster width or height exceeds the GPU's 2D texture dimension limit
 */
static std::shared_ptr<VramCache>
load_vram_cache(const std::string& input_file, const FileInfo& src_info)
{
    const int    width     = src_info.width;
    const int    height    = src_info.height;
    const int    nb        = src_info.samples_per_pixel;   // ALL physical bands
    const size_t band_px   = static_cast<size_t>(width) * height;
    const size_t band_bytes = band_px * sizeof(float);

    // Each band needs: d_bands (linear) + d_arrays (CUDA 2D copy)  → ×2
    const size_t vram_needed = band_bytes * static_cast<size_t>(nb) * 2;

    size_t free_vram = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
    const size_t vram_limit = static_cast<size_t>(free_vram * 0.80);

    if (vram_needed > vram_limit) {
        throw std::runtime_error(
            "persist() error: raster needs " +
            std::to_string(vram_needed >> 20) +
            " MB of VRAM (" + std::to_string((band_bytes * nb) >> 20) +
            " MB data + " + std::to_string((band_bytes * nb) >> 20) +
            " MB texture arrays) but only " +
            std::to_string(vram_limit >> 20) + " MB (80% of " +
            std::to_string(free_vram >> 20) + " MB free) is available. "
            "Raster: " + std::to_string(width) + "x" + std::to_string(height) +
            " x" + std::to_string(nb) + " bands @ float32.");
    }

    // Check CUDA 2D texture dimension limits.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_tex_w = 0, max_tex_h = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_tex_w, cudaDevAttrMaxTexture2DWidth,  device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_tex_h, cudaDevAttrMaxTexture2DHeight, device));
    if (width > max_tex_w || height > max_tex_h) {
        throw std::runtime_error(
            "persist() error: raster " + std::to_string(width) + "x" +
            std::to_string(height) + " exceeds CUDA 2D texture limit " +
            std::to_string(max_tex_w) + "x" + std::to_string(max_tex_h) +
            " on this GPU. Cannot persist.");
    }

    auto cache     = std::make_shared<VramCache>();
    cache->width   = width;
    cache->height  = height;
    cache->num_bands = nb;
    cache->d_bands.assign(nb, nullptr);
    cache->d_arrays.assign(nb, nullptr);
    cache->tex_bilinear.assign(nb, 0);
    cache->tex_nearest.assign(nb, 0);

    const cudaChannelFormatDesc chan = cudaCreateChannelDesc<float>();

    // ── Helper: bind a CUDA array as both bilinear and nearest texture objects ──
    auto make_textures = [&](int b) {
        // Copy d_bands[b] (linear device) → d_arrays[b] (2D array).
        CUDA_CHECK(cudaMemcpy2DToArray(
            cache->d_arrays[b], 0, 0,
            cache->d_bands[b],
            static_cast<size_t>(width) * sizeof(float),
            static_cast<size_t>(width) * sizeof(float),
            static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice));

        cudaResourceDesc res{};
        res.resType             = cudaResourceTypeArray;
        res.res.array.array     = cache->d_arrays[b];

        // Bilinear
        cudaTextureDesc td{};
        td.addressMode[0]   = cudaAddressModeClamp;
        td.addressMode[1]   = cudaAddressModeClamp;
        td.filterMode       = cudaFilterModeLinear;
        td.readMode         = cudaReadModeElementType;
        td.normalizedCoords = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cache->tex_bilinear[b], &res, &td, nullptr));

        // Nearest — same array, different filter
        td.filterMode = cudaFilterModePoint;
        CUDA_CHECK(cudaCreateTextureObject(&cache->tex_nearest[b], &res, &td, nullptr));
    };

    // ── Staging buffer: one band at a time to keep host memory pressure low ──
    float* h_stage = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_stage, band_bytes, cudaHostAllocDefault));

    const bool is_s3               = is_s3_path(input_file);
    const bool is_pixel_interleaved = (src_info.interleave == "PIXEL");

    // ── Allocate all device memory first (fail fast before any I/O) ──────────
    try {
        for (int b = 0; b < nb; ++b) {
            CUDA_CHECK(cudaMalloc(&cache->d_bands[b], band_bytes));
            CUDA_CHECK(cudaMallocArray(&cache->d_arrays[b], &chan,
                                       static_cast<size_t>(width),
                                       static_cast<size_t>(height)));
        }
    } catch (...) {
        cudaFreeHost(h_stage);
        throw;
    }

    // ── Load pixel data ───────────────────────────────────────────────────────
    try {
        if (!is_s3) {
            // ── Local path: GDAL reads one band at a time ─────────────────────
            GDALAllRegister();
            GDALDataset* ds = static_cast<GDALDataset*>(
                GDALOpen(input_file.c_str(), GA_ReadOnly));
            if (!ds) {
                throw std::runtime_error("persist(): cannot open " + input_file);
            }
            try {
                for (int b = 0; b < nb; ++b) {
                    GDALRasterBand* gdal_band = ds->GetRasterBand(b + 1);
                    if (!gdal_band)
                        throw std::runtime_error(
                            "persist(): band " + std::to_string(b + 1) + " not found.");
                    gdal_band->RasterIO(GF_Read, 0, 0, width, height,
                                        h_stage, width, height, GDT_Float32, 0, 0);
                    CUDA_CHECK(cudaMemcpy(cache->d_bands[b], h_stage,
                                          band_bytes, cudaMemcpyHostToDevice));
                    make_textures(b);
                }
            } catch (...) {
                GDALClose(ds);
                throw;
            }
            GDALClose(ds);

        } else {
            // ── S3 path: direct libcurl tile/strip fetch ───────────────────────
            S3Loc s3_loc = parse_s3_path(input_file);

            const size_t tiles_across =
                ((size_t)width  + src_info.tile_width  - 1) / src_info.tile_width;
            const size_t tiles_down   =
                ((size_t)height + src_info.tile_height - 1) / src_info.tile_height;

            // For pixel-interleaved we fetch every tile once and scatter all bands
            // simultaneously to avoid redundant S3 round-trips.
            // For band-sequential we iterate per band (tiles carry only one band).
            const int outer_loops = is_pixel_interleaved ? 1 : nb;

            // Per-band flat host buffers (full image).
            // We allocate these on the heap to avoid large stack frames.
            std::vector<std::vector<float>> band_bufs(
                nb, std::vector<float>(band_px, 0.0f));

            for (int bi_loop = 0; bi_loop < outer_loops; ++bi_loop) {
                // For band-sequential: bi_loop == physical band index.
                // For pixel-interleaved: bi_loop == 0, we fill ALL band_bufs.
                if (!is_pixel_interleaved) {
                    std::fill(band_bufs[bi_loop].begin(), band_bufs[bi_loop].end(), 0.0f);
                } else {
                    for (auto& bb : band_bufs) std::fill(bb.begin(), bb.end(), 0.0f);
                }

                if (src_info.is_tiled) {
                    std::vector<TileFetch> jobs;
                    jobs.reserve(tiles_across * tiles_down);

                    for (size_t tr = 0; tr < tiles_down; ++tr) {
                        for (size_t tc = 0; tc < tiles_across; ++tc) {
                            size_t ti;
                            if (is_pixel_interleaved) {
                                ti = tr * tiles_across + tc;
                            } else {
                                ti = static_cast<size_t>(bi_loop) * tiles_across * tiles_down
                                     + tr * tiles_across + tc;
                            }
                            if (ti < src_info.tile_offsets.size()) {
                                jobs.push_back({ti,
                                                src_info.tile_offsets[ti],
                                                src_info.tile_lengths[ti],
                                                {}, 0});
                            }
                        }
                    }

                    s3_fetch_tiles(s3_loc, src_info.tile_offsets,
                                   src_info.tile_lengths, jobs);

                    for (auto& job : jobs) {
                        if (job.err || job.data.empty()) continue;

                        const int block_spp = is_pixel_interleaved
                                              ? src_info.samples_per_pixel : 1;
                        auto ftile = decompress_tile_to_float(
                            job.data.data(), job.data.size(),
                            src_info.tile_width, src_info.tile_height,
                            block_spp, src_info.compression,
                            src_info.predictor, src_info.data_type);

                        size_t ti = job.tile_index;
                        int tr, tc;
                        if (is_pixel_interleaved) {
                            tr = static_cast<int>(ti / tiles_across);
                            tc = static_cast<int>(ti % tiles_across);
                        } else {
                            tr = static_cast<int>(
                                (ti % (tiles_across * tiles_down)) / tiles_across);
                            tc = static_cast<int>(ti % tiles_across);
                        }

                        const int tile_x0 = tc * src_info.tile_width;
                        const int tile_y0 = tr * src_info.tile_height;
                        const int copy_h  = std::min(src_info.tile_height, height - tile_y0);
                        const int copy_w  = std::min(src_info.tile_width,  width  - tile_x0);

                        // Scatter tile pixels into per-band flat buffers.
                        const int bands_to_fill = is_pixel_interleaved ? nb : 1;
                        for (int b = 0; b < bands_to_fill; ++b) {
                            const int phys = is_pixel_interleaved ? b : bi_loop;
                            for (int row = 0; row < copy_h; ++row) {
                                float* dst = band_bufs[phys].data()
                                             + static_cast<size_t>(tile_y0 + row) * width
                                             + tile_x0;
                                for (int col = 0; col < copy_w; ++col) {
                                    const int src_px = row * src_info.tile_width + col;
                                    dst[col] = is_pixel_interleaved
                                        ? ftile[src_px * block_spp + b]
                                        : ftile[src_px];
                                }
                            }
                        }
                    }

                } else {
                    // Strip-organised S3 file
                    const int strips_per_band =
                        (height + src_info.rows_per_strip - 1) / src_info.rows_per_strip;

                    std::vector<TileFetch> jobs;
                    for (int sr = 0; sr < strips_per_band; ++sr) {
                        size_t si = is_pixel_interleaved
                            ? static_cast<size_t>(sr)
                            : static_cast<size_t>(bi_loop) * strips_per_band + sr;
                        if (si < src_info.strip_offsets.size()) {
                            jobs.push_back({si,
                                            src_info.strip_offsets[si],
                                            src_info.strip_lengths[si],
                                            {}, 0});
                        }
                    }

                    s3_fetch_tiles(s3_loc, src_info.strip_offsets,
                                   src_info.strip_lengths, jobs);

                    for (auto& job : jobs) {
                        if (job.err || job.data.empty()) continue;

                        int sr = static_cast<int>(job.tile_index);
                        if (!is_pixel_interleaved) sr %= strips_per_band;

                        const int strip_y0    = sr * src_info.rows_per_strip;
                        const int strip_rows  = std::min(src_info.rows_per_strip,
                                                         height - strip_y0);
                        const int block_spp   = is_pixel_interleaved
                                                ? src_info.samples_per_pixel : 1;
                        auto fstrip = decompress_tile_to_float(
                            job.data.data(), job.data.size(),
                            width, strip_rows, block_spp,
                            src_info.compression, src_info.predictor,
                            src_info.data_type);

                        const int bands_to_fill = is_pixel_interleaved ? nb : 1;
                        for (int b = 0; b < bands_to_fill; ++b) {
                            const int phys = is_pixel_interleaved ? b : bi_loop;
                            for (int row = 0; row < strip_rows; ++row) {
                                float* dst = band_bufs[phys].data()
                                             + static_cast<size_t>(strip_y0 + row) * width;
                                if (is_pixel_interleaved) {
                                    for (int x = 0; x < width; ++x)
                                        dst[x] = fstrip[
                                            static_cast<size_t>(row) * width * block_spp
                                            + x * block_spp + b];
                                } else {
                                    memcpy(dst,
                                           fstrip.data() +
                                               static_cast<size_t>(row) * width,
                                           static_cast<size_t>(width) * sizeof(float));
                                }
                            }
                        }
                    }
                }

                // Upload assembled band(s) to VRAM.
                const int upload_start = is_pixel_interleaved ? 0       : bi_loop;
                const int upload_end   = is_pixel_interleaved ? nb - 1  : bi_loop;
                for (int b = upload_start; b <= upload_end; ++b) {
                    memcpy(h_stage, band_bufs[b].data(), band_bytes);
                    CUDA_CHECK(cudaMemcpy(cache->d_bands[b], h_stage,
                                          band_bytes, cudaMemcpyHostToDevice));
                    make_textures(b);
                }
            } // end outer_loops
        }

    } catch (...) {
        cudaFreeHost(h_stage);
        throw;
    }
    cudaFreeHost(h_stage);

    // ── Build device-side texture handle arrays ───────────────────────────────
    const size_t tex_arr_bytes = static_cast<size_t>(nb) * sizeof(cudaTextureObject_t);
    CUDA_CHECK(cudaMalloc(&cache->d_tex_bilinear_dev, tex_arr_bytes));
    CUDA_CHECK(cudaMemcpy(cache->d_tex_bilinear_dev, cache->tex_bilinear.data(),
                          tex_arr_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&cache->d_tex_nearest_dev, tex_arr_bytes));
    CUDA_CHECK(cudaMemcpy(cache->d_tex_nearest_dev, cache->tex_nearest.data(),
                          tex_arr_bytes, cudaMemcpyHostToDevice));

    return cache;
}

// ── Chain::persist() ─────────────────────────────────────────────────────────

std::shared_ptr<Chain> Chain::persist()
{
    FileInfo src_info = get_file_info(input_file_);
    auto cache = load_vram_cache(input_file_, src_info);

    auto new_chain   = std::make_shared<Chain>(*this);
    new_chain->cache_ = std::move(cache);
    return new_chain;
}



bool Chain::has_reproject_operation() const {
    for (const auto& op : operations_) {
        if (op.type == ChainOpType::REPROJECT) { return true; }
    }
    return false;
}

FileInfo Chain::get_output_info() const {
    FileInfo src_info = get_file_info(input_file_);
    for (const auto& op : operations_) {
        if (op.type == ChainOpType::REPROJECT) {
            return pre_pass_reproject(input_file_, src_info, op.reproject_params);
        }
    }
    return src_info;
}



static PipelineCtx build_pipeline_context(const std::vector<ChainOp>& operations,
                                           const FileInfo&              src_info,
                                           const std::string&           input_file,
                                           const std::vector<int>&      user_band_selection = {}) {
    PipelineCtx ctx;

    for (const auto& op : operations) {
        if (op.type == ChainOpType::REPROJECT) {
            ctx.has_reproject        = true;
            ctx.reproject_params     = op.reproject_params;
            ctx.reproject_output_info = pre_pass_reproject(input_file, src_info,
                                                            op.reproject_params);
            break;
        }
    }

    const FileInfo& clip_reference_info = ctx.has_reproject
                                        ? ctx.reproject_output_info
                                        : src_info;

    std::vector<int> band_map;
    bool has_algebra       = false;
    bool has_neighborhood  = false;

    for (const auto& op : operations) {
        if (op.type == ChainOpType::ALGEBRA) {
            ctx.instructions = compile_algebra_expression(op.algebra_expr, band_map);
            has_algebra = true;
        }
        if (op.type == ChainOpType::CLIP) {
            ctx.has_clip_mask = true;
            parse_polygon_to_spans(op.geojson_str, clip_reference_info, ctx.clip_spans);
        }
        if (op.type == ChainOpType::FOCAL) {
            ctx.has_focal   = true;
            ctx.focal_params = op.focal_params;
            ctx.focal_num_output_bands = 1;
            has_neighborhood = true;
        }
        if (op.type == ChainOpType::TERRAIN) {
            ctx.has_terrain    = true;
            ctx.terrain_params = op.terrain_params;
            ctx.focal_num_output_bands = op.terrain_params.num_output_bands;
            has_neighborhood = true;
        }
        if (op.type == ChainOpType::TEXTURE) {
            ctx.has_texture  = true;
            ctx.glcm_params  = op.glcm_params;
            ctx.focal_num_output_bands = op.glcm_params.num_output_bands;
            has_neighborhood = true;
        }
        if (op.type == ChainOpType::ZONAL_STATS) {
            ctx.has_zonal    = true;
            ctx.zonal_params = op.zonal_params;
        }
    }

    if (band_map.empty()) {
        if (!user_band_selection.empty()) {
            band_map = user_band_selection;
        } else {
            for (int b = 0; b < src_info.samples_per_pixel; ++b) band_map.push_back(b);
        }
    }

    ctx.band_map = band_map;
    return ctx;
}



// Helper: build and attach the result/queue callbacks to ctx.
static void attach_output_callbacks(PipelineCtx& ctx,
                                    GDALRasterBand* output_band,
                                    RasterResult*   result,
                                    std::shared_ptr<ChunkQueue> chunk_queue,
                                    int num_out_bands) {
    if (output_band) ctx.output_band = output_band;
    if (result) {
        int tw = result->width, th = result->height, tb = result->bands;
        ctx.result_callback = [result, tw, th, tb](int w, int h, float* px, int y0) {
            size_t cp = (size_t)w * h, tp = (size_t)tw * th;
            for (int b = 0; b < tb; ++b)
                memcpy(result->data.data() + b * tp + (size_t)y0 * w,
                       px + b * cp, cp * sizeof(float));
        };
    }
    if (chunk_queue) {
        ctx.queue_callback = [chunk_queue, num_out_bands](int w, int h, float* px, int y0) {
            ChunkResult c; c.width = w; c.height = h; c.y_offset = y0;
            c.data.assign(px, px + (size_t)w * h * num_out_bands);
            chunk_queue->push(std::move(c));
        };
    }
}

void Chain::execute(GDALRasterBand*             output_band,
                    RasterResult*               result,
                    std::shared_ptr<ChunkQueue> chunk_queue,
                    bool                        verbose) {
    static std::atomic<int> vsimem_counter{0};

    // ── locate the first and last neighborhood op ──────────────────────────
    int first_nbhd = -1, last_nbhd = -1;
    for (int i = 0; i < (int)operations_.size(); ++i) {
        auto t = operations_[i].type;
        if (t == ChainOpType::FOCAL || t == ChainOpType::TERRAIN || t == ChainOpType::TEXTURE) {
            if (first_nbhd < 0) first_nbhd = i;
            last_nbhd = i;
        }
    }

    // ── no neighborhood op: algebra/passthrough/zonal path ─────────────────
    if (first_nbhd < 0) {
        FileInfo src_info = get_file_info(input_file_);

        // Find first REPROJECT.  If any op precedes it (clip, algebra) it must
        // run in source CRS first; the reproject+post then runs on the result.
        int reproject_idx = -1;
        for (int i = 0; i < (int)operations_.size(); ++i)
            if (operations_[i].type == ChainOpType::REPROJECT) { reproject_idx = i; break; }

        if (reproject_idx > 0) {
            // ── split: pre-reproject pass (source CRS) → vsimem, then reproject+post ──
            std::vector<ChainOp> pre_ops (operations_.begin(), operations_.begin() + reproject_idx);
            std::vector<ChainOp> post_ops(operations_.begin() + reproject_idx, operations_.end());

            PipelineCtx pre_ctx = build_pipeline_context(pre_ops, src_info, input_file_, user_band_selection_);
            pre_ctx.vram_cache = cache_;
            int pre_bands = pre_ctx.instructions.empty() ? (int)pre_ctx.band_map.size() : 1;

            std::string split_vsimem = "/vsimem/curaster_split_"
                                     + std::to_string(getpid()) + "_"
                                     + std::to_string(++vsimem_counter) + ".tif";
            GDALDataset* split_ds = create_output_dataset(split_vsimem, src_info, pre_bands);
            if (pre_bands == 1)
                pre_ctx.output_band = split_ds->GetRasterBand(1);
            else
                pre_ctx.output_dataset = static_cast<void*>(split_ds);
            try {
                run_engine_ex(input_file_, pre_ctx, false);
            } catch (...) {
                GDALClose(split_ds); VSIUnlink(split_vsimem.c_str()); throw;
            }
            GDALClose(split_ds);

            FileInfo split_info = get_file_info(split_vsimem);
            PipelineCtx post_ctx = build_pipeline_context(post_ops, split_info, split_vsimem);
            int num_out = post_ctx.instructions.empty() ? (int)post_ctx.band_map.size() : 1;
            attach_output_callbacks(post_ctx, output_band, result, chunk_queue, num_out);
            try {
                run_engine_ex(split_vsimem, post_ctx, verbose);
            } catch (...) {
                VSIUnlink(split_vsimem.c_str()); throw;
            }
            VSIUnlink(split_vsimem.c_str());
        } else {
            // No pre-reproject ops: single pass
            PipelineCtx ctx = build_pipeline_context(operations_, src_info, input_file_, user_band_selection_);
            ctx.vram_cache = cache_;
            int num_out = ctx.instructions.empty() ? (int)ctx.band_map.size() : 1;
            attach_output_callbacks(ctx, output_band, result, chunk_queue, num_out);
            if (ctx.has_zonal) run_engine_zonal(input_file_, ctx, verbose);
            else                run_engine_ex  (input_file_, ctx, verbose);
        }
        return;
    }

    // ── split into pre | neighborhood | post ───────────────────────────────
    std::vector<ChainOp> pre_ops (operations_.begin(),               operations_.begin() + first_nbhd);
    std::vector<ChainOp> nbhd_ops(operations_.begin() + first_nbhd,  operations_.begin() + last_nbhd + 1);
    std::vector<ChainOp> post_ops(operations_.begin() + last_nbhd + 1, operations_.end());

    std::string effective_input = input_file_;
    std::string pre_vsimem, focal_vsimem;

    auto do_cleanup = [&]() {
        if (!pre_vsimem.empty())   { VSIUnlink(pre_vsimem.c_str());   pre_vsimem.clear(); }
        if (!focal_vsimem.empty()) { VSIUnlink(focal_vsimem.c_str()); focal_vsimem.clear(); }
    };

    try {
        // ── pre-pass: run clip/reproject/algebra before the neighborhood op ──
        if (!pre_ops.empty()) {
            FileInfo pre_src = get_file_info(input_file_);
            PipelineCtx pre_ctx = build_pipeline_context(pre_ops, pre_src, input_file_, user_band_selection_);
            pre_ctx.vram_cache = cache_;
            const FileInfo& pre_out =
                pre_ctx.has_reproject ? pre_ctx.reproject_output_info : pre_src;
            int pre_out_bands = pre_ctx.instructions.empty() ? (int)pre_ctx.band_map.size() : 1;

            pre_vsimem = "/vsimem/curaster_pre_"
                       + std::to_string(getpid()) + "_"
                       + std::to_string(++vsimem_counter) + ".tif";
            GDALDataset* pre_ds = create_output_dataset(pre_vsimem, pre_out, pre_out_bands);
            if (pre_out_bands == 1) {
                pre_ctx.output_band = pre_ds->GetRasterBand(1);
            } else {
                pre_ctx.output_dataset = static_cast<void*>(pre_ds);
            }
            run_engine_ex(input_file_, pre_ctx, false);
            GDALClose(pre_ds);
            effective_input = pre_vsimem;
        }

        // ── neighborhood pass ───────────────────────────────────────────────
        FileInfo nbhd_src = get_file_info(effective_input);
        PipelineCtx nbhd_ctx = build_pipeline_context(nbhd_ops, nbhd_src, effective_input);
        int focal_out_bands = nbhd_ctx.focal_num_output_bands;
        if (focal_out_bands <= 0) focal_out_bands = 1;

        const bool multiband_focal = nbhd_ctx.has_focal
            && !nbhd_ctx.has_terrain
            && !nbhd_ctx.has_texture
            && (int)nbhd_ctx.band_map.size() > 1;

        if (multiband_focal) {
            int N = (int)nbhd_ctx.band_map.size();
            focal_vsimem = "/vsimem/curaster_focal_"
                         + std::to_string(getpid()) + "_"
                         + std::to_string(++vsimem_counter) + ".tif";
            GDALDataset* focal_ds = create_output_dataset(focal_vsimem, nbhd_src, N);
            try {
                for (int b = 0; b < N; ++b) {
                    PipelineCtx bc;
                    bc.has_focal              = true;
                    bc.focal_params           = nbhd_ctx.focal_params;
                    bc.focal_num_output_bands = 1;
                    bc.band_map               = { nbhd_ctx.band_map[b] };
                    bc.has_clip_mask          = nbhd_ctx.has_clip_mask;
                    bc.clip_spans             = nbhd_ctx.clip_spans;
                    bc.output_band            = focal_ds->GetRasterBand(b + 1);
                    run_engine_focal(effective_input, bc, false);
                }
            } catch (...) {
                GDALClose(focal_ds);
                VSIUnlink(focal_vsimem.c_str());
                focal_vsimem.clear();
                throw;
            }
            GDALClose(focal_ds);

            if (!post_ops.empty()) {
                FileInfo post_src = get_file_info(focal_vsimem);
                PipelineCtx post_ctx = build_pipeline_context(post_ops, post_src, focal_vsimem);
                // post_ctx.instructions non-empty only if post_ops contain ALGEBRA → 1 band out;
                // otherwise it's passthrough/clip/reproject and all N bands flow through.
                int post_out = post_ctx.instructions.empty() ? N : 1;
                attach_output_callbacks(post_ctx, output_band, result, chunk_queue, post_out);
                run_engine_ex(focal_vsimem, post_ctx, verbose);
            } else {
                FileInfo fvs = get_file_info(focal_vsimem);
                PipelineCtx out_ctx = build_pipeline_context({}, fvs, focal_vsimem);
                attach_output_callbacks(out_ctx, output_band, result, chunk_queue, N);
                run_engine_ex(focal_vsimem, out_ctx, verbose);
            }

        } else if (!post_ops.empty()) {
            // Write focal output to a vsimem file so post-ops can consume it.
            focal_vsimem = "/vsimem/curaster_focal_"
                         + std::to_string(getpid()) + "_"
                         + std::to_string(++vsimem_counter) + ".tif";
            GDALDataset* focal_ds = create_output_dataset(focal_vsimem, nbhd_src, focal_out_bands);
            if (focal_out_bands == 1)
                nbhd_ctx.output_band    = focal_ds->GetRasterBand(1);
            else
                nbhd_ctx.output_dataset = static_cast<void*>(focal_ds);
            run_engine_focal(effective_input, nbhd_ctx, verbose);
            GDALClose(focal_ds);

            // ── post-pass: algebra applied to focal output ──────────────────
            FileInfo post_src = get_file_info(focal_vsimem);
            PipelineCtx post_ctx = build_pipeline_context(post_ops, post_src, focal_vsimem);
            int post_out = 1;
            attach_output_callbacks(post_ctx, output_band, result, chunk_queue, post_out);
            run_engine_ex(focal_vsimem, post_ctx, verbose);

        } else {
            // Direct output from the neighborhood pass.
            attach_output_callbacks(nbhd_ctx, output_band, result, chunk_queue, focal_out_bands);
            run_engine_focal(effective_input, nbhd_ctx, verbose);
        }
    } catch (...) {
        do_cleanup();
        throw;
    }
    do_cleanup();
}

std::shared_ptr<Chain> Chain::algebra(const std::string& expression) {
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::ALGEBRA;
    op.algebra_expr = expression;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::select_bands(std::vector<int> bands) {
    auto c = std::make_shared<Chain>(*this);
    for (int& b : bands) b--;
    c->user_band_selection_ = std::move(bands);
    return c;
}

std::shared_ptr<Chain> Chain::clip(const std::string& geojson) {
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::CLIP;
    op.geojson_str = geojson;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::reproject(const std::string& target_crs,
                                         double             res_x,
                                         double             res_y,
                                         const std::string& resampling,
                                         double             nodata,
                                         double             te_xmin,
                                         double             te_ymin,
                                         double             te_xmax,
                                         double             te_ymax) {
    ReprojectParams rp;
    rp.target_crs   = target_crs;
    rp.pixel_size_x = res_x;
    rp.pixel_size_y = res_y;
    rp.resampling   = (resampling == "nearest")
                        ? ResampleMethod::NEAREST
                        : ResampleMethod::BILINEAR;
    rp.nodata_value = nodata;

    if (te_xmin != 0.0 || te_xmax != 0.0) {
        rp.has_extent  = true;
        rp.extent_xmin = te_xmin;
        rp.extent_ymin = te_ymin;
        rp.extent_xmax = te_xmax;
        rp.extent_ymax = te_ymax;
    }

    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::REPROJECT;
    op.reproject_params = rp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::focal(const std::string& stat,
                                     int radius,
                                     const std::string& shape,
                                     bool clamp_border)
{
    FocalParams fp;
    build_focal_params(fp, stat, radius, shape, clamp_border);
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::FOCAL;
    op.focal_params = fp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::terrain(const std::vector<std::string>& metrics,
                                       const std::string& unit,
                                       double sun_azimuth,
                                       double sun_altitude,
                                       const std::string& method)
{
    FileInfo src_info = get_file_info(input_file_);
    TerrainParams tp;
    build_terrain_params(tp, metrics.empty() ? std::vector<std::string>{"slope"} : metrics,
                         unit, sun_azimuth, sun_altitude, method, src_info);
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::TERRAIN;
    op.terrain_params = tp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::texture(const std::vector<std::string>& features,
                                       int window,
                                       int levels,
                                       const std::string& direction_mode,
                                       bool log_scale,
                                       float val_min,
                                       float val_max)
{
    GLCMParams gp;
    build_glcm_params(gp, features, window, levels, direction_mode, log_scale, val_min, val_max);
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::TEXTURE;
    op.glcm_params = gp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

// Build a single-feature GeoJSON FeatureCollection covering the full raster extent.
static std::string make_bbox_geojson(const FileInfo& info) {
    double x0 = info.geo_transform[0];
    double y0 = info.geo_transform[3];
    double x1 = x0 + info.geo_transform[1] * info.width;
    double y1 = y0 + info.geo_transform[5] * info.height;
    double xmin = x0 < x1 ? x0 : x1,  xmax = x0 < x1 ? x1 : x0;
    double ymin = y0 < y1 ? y0 : y1,  ymax = y0 < y1 ? y1 : y0;
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\","
        "\"geometry\":{\"type\":\"Polygon\",\"coordinates\":[["
        "[%.15g,%.15g],[%.15g,%.15g],[%.15g,%.15g],[%.15g,%.15g],[%.15g,%.15g]"
        "]]},\"properties\":{\"id\":1}}]}",
        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin);
    return buf;
}

std::vector<ZoneResult> Chain::zonal_stats(const std::vector<std::string>& stats,
                                             int band,
                                             const std::string& geojson_str,
                                             bool verbose)
{
    static std::atomic<int> zonal_counter{0};

    std::string effective_input = input_file_;
    std::string pre_vsimem;

    if (!operations_.empty()) {
        FileInfo pre_src = get_file_info(input_file_);
        PipelineCtx pre_ctx = build_pipeline_context(operations_, pre_src, input_file_, user_band_selection_);
        pre_ctx.vram_cache = cache_;

        int pre_out_bands;
        if (pre_ctx.has_focal || pre_ctx.has_terrain || pre_ctx.has_texture)
            pre_out_bands = pre_ctx.focal_num_output_bands > 0 ? pre_ctx.focal_num_output_bands : 1;
        else
            pre_out_bands = pre_ctx.instructions.empty() ? (int)pre_ctx.band_map.size() : 1;

        const FileInfo& pre_out_info =
            pre_ctx.has_reproject ? pre_ctx.reproject_output_info : pre_src;

        pre_vsimem = "/vsimem/curaster_zonal_pre_"
                   + std::to_string(getpid()) + "_"
                   + std::to_string(++zonal_counter) + ".tif";
        GDALDataset* pre_ds = create_output_dataset(pre_vsimem, pre_out_info, pre_out_bands);

        try {
            if (pre_out_bands == 1) {
                pre_ctx.output_band = pre_ds->GetRasterBand(1);
                if (pre_ctx.has_focal || pre_ctx.has_terrain || pre_ctx.has_texture)
                    run_engine_focal(input_file_, pre_ctx, false);
                else
                    run_engine_ex(input_file_, pre_ctx, false);
            } else {
                pre_ctx.output_dataset = static_cast<void*>(pre_ds);
                if (pre_ctx.has_focal || pre_ctx.has_terrain || pre_ctx.has_texture)
                    run_engine_focal(input_file_, pre_ctx, false);
                else
                    run_engine_ex(input_file_, pre_ctx, false);
            }
        } catch (...) {
            GDALClose(pre_ds);
            VSIUnlink(pre_vsimem.c_str());
            throw;
        }
        GDALClose(pre_ds);
        effective_input = pre_vsimem;
    }

    // Get geometry of the effective input (after any preceding ops).
    FileInfo eff_info = get_file_info(effective_input);

    // If no AOI provided, compute stats over the whole raster extent.
    std::string effective_geojson = geojson_str.empty()
        ? make_bbox_geojson(eff_info)
        : geojson_str;

    // Build a zonal-only pipeline and run it.
    ChainOp zop;
    zop.type                     = ChainOpType::ZONAL_STATS;
    zop.zonal_params.geojson_str = effective_geojson;
    zop.zonal_params.stats       = stats;
    zop.zonal_params.band        = band;

    std::vector<ChainOp> zonal_ops = {zop};
    PipelineCtx ctx = build_pipeline_context(zonal_ops, eff_info, effective_input);

    try {
        run_engine_zonal(effective_input, ctx, verbose);
    } catch (...) {
        if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
        throw;
    }
    if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
    return ctx.zonal_results;
}




void Chain::save_local(const std::string& output_path, bool verbose) {
    static std::atomic<int> sl_counter{0};

    int first_nbhd = -1, last_nbhd = -1;
    for (int i = 0; i < (int)operations_.size(); ++i) {
        auto t = operations_[i].type;
        if (t == ChainOpType::FOCAL || t == ChainOpType::TERRAIN || t == ChainOpType::TEXTURE) {
            if (first_nbhd < 0) first_nbhd = i;
            last_nbhd = i;
        }
    }
    bool has_post_ops = (last_nbhd >= 0 && last_nbhd + 1 < (int)operations_.size());

    // Algebra anywhere in the chain collapses output to 1 band.
    // Check post-neighborhood ops separately so CLIP/REPROJECT post-ops
    // don't incorrectly force single-band output for multiband focal chains.
    bool post_has_algebra = false;
    for (int i = last_nbhd + 1; i < (int)operations_.size(); ++i)
        if (operations_[i].type == ChainOpType::ALGEBRA) { post_has_algebra = true; break; }

    FileInfo src_info = get_file_info(input_file_);
    PipelineCtx peek = build_pipeline_context(operations_, src_info, input_file_, user_band_selection_);

    int num_out_bands;
    if (!peek.instructions.empty() || post_has_algebra) {
        // Algebra anywhere → 1-band output
        num_out_bands = 1;
    } else if (first_nbhd < 0) {
        num_out_bands = (int)peek.band_map.size();
    } else if (peek.has_focal && !peek.has_terrain && !peek.has_texture) {
        num_out_bands = (int)peek.band_map.size();
    } else {
        num_out_bands = peek.focal_num_output_bands > 0 ? peek.focal_num_output_bands : 1;
    }

    FileInfo out_info = get_output_info();
    GDALDataset* output_ds = create_output_dataset(output_path, out_info, num_out_bands);

    try {
        if (num_out_bands == 1) {
            execute(output_ds->GetRasterBand(1), nullptr, nullptr, verbose);

        } else if (first_nbhd < 0) {
            // Mirror the execute() split: if any op precedes REPROJECT, run it
            // in source CRS first so that clip geometry is interpreted correctly.
            int reproject_idx = -1;
            for (int i = 0; i < (int)operations_.size(); ++i)
                if (operations_[i].type == ChainOpType::REPROJECT) { reproject_idx = i; break; }

            if (reproject_idx > 0) {
                std::vector<ChainOp> pre_ops (operations_.begin(), operations_.begin() + reproject_idx);
                std::vector<ChainOp> post_ops(operations_.begin() + reproject_idx, operations_.end());

                PipelineCtx pre_ctx = build_pipeline_context(pre_ops, src_info, input_file_, user_band_selection_);
                pre_ctx.vram_cache = cache_;
                int pre_bands = pre_ctx.instructions.empty() ? (int)pre_ctx.band_map.size() : 1;

                std::string split_vsimem = "/vsimem/curaster_sl_split_"
                                         + std::to_string(getpid()) + "_"
                                         + std::to_string(++sl_counter) + ".tif";
                GDALDataset* split_ds = create_output_dataset(split_vsimem, src_info, pre_bands);
                if (pre_bands == 1)
                    pre_ctx.output_band = split_ds->GetRasterBand(1);
                else
                    pre_ctx.output_dataset = static_cast<void*>(split_ds);
                try {
                    run_engine_ex(input_file_, pre_ctx, false);
                } catch (...) {
                    GDALClose(split_ds); VSIUnlink(split_vsimem.c_str()); throw;
                }
                GDALClose(split_ds);

                FileInfo split_info = get_file_info(split_vsimem);
                PipelineCtx post_ctx = build_pipeline_context(post_ops, split_info, split_vsimem);
                post_ctx.output_dataset = static_cast<void*>(output_ds);
                try {
                    run_engine_ex(split_vsimem, post_ctx, verbose);
                } catch (...) {
                    VSIUnlink(split_vsimem.c_str()); throw;
                }
                VSIUnlink(split_vsimem.c_str());
            } else {
                PipelineCtx ctx = build_pipeline_context(operations_, src_info, input_file_, user_band_selection_);
                ctx.vram_cache = cache_;
                ctx.output_dataset = static_cast<void*>(output_ds);
                run_engine_ex(input_file_, ctx, verbose);
            }

        } else {
            std::string effective_input = input_file_;
            std::string pre_vsimem;

            if (first_nbhd > 0) {
                std::vector<ChainOp> pre_ops(operations_.begin(), operations_.begin() + first_nbhd);
                FileInfo pre_src = get_file_info(input_file_);
                PipelineCtx pre_ctx = build_pipeline_context(pre_ops, pre_src, input_file_, user_band_selection_);
                pre_ctx.vram_cache = cache_;
                const FileInfo& pre_out = pre_ctx.has_reproject ? pre_ctx.reproject_output_info : pre_src;
                int pre_out_bands = pre_ctx.instructions.empty() ? (int)pre_ctx.band_map.size() : 1;

                pre_vsimem = "/vsimem/curaster_sl_pre_"
                           + std::to_string(getpid()) + "_"
                           + std::to_string(++sl_counter) + ".tif";
                GDALDataset* pre_ds = create_output_dataset(pre_vsimem, pre_out, pre_out_bands);
                if (pre_out_bands == 1)
                    pre_ctx.output_band = pre_ds->GetRasterBand(1);
                else
                    pre_ctx.output_dataset = static_cast<void*>(pre_ds);
                run_engine_ex(input_file_, pre_ctx, false);
                GDALClose(pre_ds);
                effective_input = pre_vsimem;
            }

            FileInfo nbhd_src = get_file_info(effective_input);

            // Ops after the last neighborhood op (may include CLIP and/or REPROJECT).
            std::vector<ChainOp> post_focal_ops(operations_.begin() + last_nbhd + 1, operations_.end());

            // Index of REPROJECT within post_focal_ops (-1 if none).
            int post_reproject_idx = -1;
            for (int i = 0; i < (int)post_focal_ops.size(); ++i)
                if (post_focal_ops[i].type == ChainOpType::REPROJECT && post_reproject_idx < 0)
                    post_reproject_idx = i;

            // Full context (nbhd → end) gives us focal_params, band_map, and any
            // clip spans.  We'll selectively apply them below.
            std::vector<ChainOp> focal_ops_full(operations_.begin() + first_nbhd, operations_.end());
            PipelineCtx focal_ctx = build_pipeline_context(focal_ops_full, nbhd_src, effective_input);

            // Helper: build context for the focal pass including any CLIPs that
            // come BEFORE the reproject (they should be applied in source CRS).
            auto make_focal_only_ctx = [&]() -> PipelineCtx {
                std::vector<ChainOp> ops;
                ops.insert(ops.end(),
                    operations_.begin() + first_nbhd,
                    operations_.begin() + last_nbhd + 1);
                if (post_reproject_idx > 0) {
                    for (int i = 0; i < post_reproject_idx; ++i)
                        if (post_focal_ops[i].type == ChainOpType::CLIP)
                            ops.push_back(post_focal_ops[i]);
                }
                return build_pipeline_context(ops, nbhd_src, effective_input);
            };

            if (focal_ctx.has_focal && !focal_ctx.has_terrain && !focal_ctx.has_texture
                    && (int)focal_ctx.band_map.size() > 1) {

                if (post_reproject_idx >= 0) {
                    // ── Two-stage: N focal passes → tmp vsimem, then reproject+post → output_ds ──
                    PipelineCtx focal_only = make_focal_only_ctx();

                    std::string focal_tmp = "/vsimem/curaster_sl_focal_"
                                         + std::to_string(getpid()) + "_"
                                         + std::to_string(++sl_counter) + ".tif";
                    GDALDataset* focal_tmp_ds = create_output_dataset(focal_tmp, nbhd_src, num_out_bands);
                    try {
                        for (int b = 0; b < num_out_bands; ++b) {
                            PipelineCtx bc;
                            bc.has_focal              = true;
                            bc.focal_params           = focal_only.focal_params;
                            bc.focal_num_output_bands = 1;
                            bc.band_map               = { focal_only.band_map[b] };
                            bc.has_clip_mask          = focal_only.has_clip_mask;
                            bc.clip_spans             = focal_only.clip_spans;
                            bc.output_band            = focal_tmp_ds->GetRasterBand(b + 1);
                            run_engine_focal(effective_input, bc, false);
                        }
                    } catch (...) {
                        GDALClose(focal_tmp_ds); VSIUnlink(focal_tmp.c_str());
                        if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
                        throw;
                    }
                    GDALClose(focal_tmp_ds);

                    // Build post_focal_ops without the pre-reproject CLIPs that
                    // were already applied above.
                    std::vector<ChainOp> post_stripped;
                    for (int i = 0; i < (int)post_focal_ops.size(); ++i) {
                        if (i < post_reproject_idx && post_focal_ops[i].type == ChainOpType::CLIP)
                            continue;
                        post_stripped.push_back(post_focal_ops[i]);
                    }
                    FileInfo focal_tmp_info = get_file_info(focal_tmp);
                    PipelineCtx post_ctx = build_pipeline_context(post_stripped, focal_tmp_info, focal_tmp);
                    post_ctx.output_dataset = static_cast<void*>(output_ds);
                    try {
                        run_engine_ex(focal_tmp, post_ctx, verbose);
                    } catch (...) {
                        VSIUnlink(focal_tmp.c_str());
                        if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
                        throw;
                    }
                    VSIUnlink(focal_tmp.c_str());

                } else {
                    // No post-reproject: N focal passes directly to output_ds.
                    // Clip (if any) is applied via focal_ctx spans during each focal pass.
                    for (int b = 0; b < num_out_bands; ++b) {
                        PipelineCtx bc;
                        bc.has_focal              = true;
                        bc.focal_params           = focal_ctx.focal_params;
                        bc.focal_num_output_bands = 1;
                        bc.band_map               = { focal_ctx.band_map[b] };
                        bc.has_clip_mask          = focal_ctx.has_clip_mask;
                        bc.clip_spans             = focal_ctx.clip_spans;
                        bc.output_band            = output_ds->GetRasterBand(b + 1);
                        run_engine_focal(effective_input, bc, verbose && b == num_out_bands - 1);
                    }
                }

            } else {
                // Terrain / texture / single-band focal.
                if (post_reproject_idx >= 0) {
                    // Two-stage: focal → tmp vsimem, then reproject+post → output_ds.
                    PipelineCtx focal_only = make_focal_only_ctx();
                    int focal_n = focal_only.focal_num_output_bands > 0 ? focal_only.focal_num_output_bands : 1;

                    std::string focal_tmp = "/vsimem/curaster_sl_focal_"
                                         + std::to_string(getpid()) + "_"
                                         + std::to_string(++sl_counter) + ".tif";
                    GDALDataset* focal_tmp_ds = create_output_dataset(focal_tmp, nbhd_src, focal_n);
                    if (focal_n == 1)
                        focal_only.output_band = focal_tmp_ds->GetRasterBand(1);
                    else
                        focal_only.output_dataset = static_cast<void*>(focal_tmp_ds);
                    try {
                        run_engine_focal(effective_input, focal_only, false);
                    } catch (...) {
                        GDALClose(focal_tmp_ds); VSIUnlink(focal_tmp.c_str());
                        if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
                        throw;
                    }
                    GDALClose(focal_tmp_ds);

                    std::vector<ChainOp> post_stripped;
                    for (int i = 0; i < (int)post_focal_ops.size(); ++i) {
                        if (i < post_reproject_idx && post_focal_ops[i].type == ChainOpType::CLIP)
                            continue;
                        post_stripped.push_back(post_focal_ops[i]);
                    }
                    FileInfo focal_tmp_info = get_file_info(focal_tmp);
                    PipelineCtx post_ctx = build_pipeline_context(post_stripped, focal_tmp_info, focal_tmp);
                    post_ctx.output_dataset = static_cast<void*>(output_ds);
                    try {
                        run_engine_ex(focal_tmp, post_ctx, verbose);
                    } catch (...) {
                        VSIUnlink(focal_tmp.c_str());
                        if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
                        throw;
                    }
                    VSIUnlink(focal_tmp.c_str());
                } else {
                    focal_ctx.output_dataset = static_cast<void*>(output_ds);
                    run_engine_focal(effective_input, focal_ctx, verbose);
                }
            }

            if (!pre_vsimem.empty()) VSIUnlink(pre_vsimem.c_str());
        }
    } catch (...) {
        GDALClose(output_ds);
        throw;
    }
    GDALClose(output_ds);
}

void Chain::save_s3(const std::string& s3_path, bool verbose) {
    // Write to a real disk temp file first (chunk-by-chunk, no RAM accumulation),
    // then upload to S3 as a single sequential copy.  This avoids GDAL buffering
    // the whole output in /vsimem/ when CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES.
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    std::string tmp_path = std::string(tmp_dir) + "curaster_s3_"
                         + std::to_string(GetCurrentProcessId()) + ".tif";
#else
    std::string tmp_path = std::string("/tmp/curaster_s3_") + std::to_string(getpid()) + ".tif";
#endif

    try {
        save_local(tmp_path, verbose);
    } catch (...) {
        VSIUnlink(tmp_path.c_str());
        throw;
    }

    if (verbose) {
        printf("\n[save_s3] Uploading %s → %s\n", tmp_path.c_str(), s3_path.c_str());
        fflush(stdout);
    }

    if (CPLCopyFile(s3_path.c_str(), tmp_path.c_str()) != 0) {
        VSIUnlink(tmp_path.c_str());
        throw std::runtime_error(
            "S3 upload failed. Ensure the path starts with /vsis3/ and AWS credentials are valid.");
    }
    VSIUnlink(tmp_path.c_str());
}

std::shared_ptr<RasterResult> Chain::to_memory(bool verbose) {
    FileInfo src_info  = get_file_info(input_file_);
    PipelineCtx tmp_ctx = build_pipeline_context(operations_, src_info, input_file_, user_band_selection_);
    int num_out_bands;
    if (!tmp_ctx.has_focal && !tmp_ctx.has_terrain && !tmp_ctx.has_texture) {
        num_out_bands = tmp_ctx.instructions.empty() ? (int)tmp_ctx.band_map.size() : 1;
    } else if (tmp_ctx.has_focal && !tmp_ctx.has_terrain && !tmp_ctx.has_texture) {
        num_out_bands = (int)tmp_ctx.band_map.size();
    } else {
        num_out_bands = tmp_ctx.focal_num_output_bands > 0 ? tmp_ctx.focal_num_output_bands : 1;
    }

    FileInfo out_info = get_output_info();

    size_t required_bytes = static_cast<size_t>(out_info.width)
                          * static_cast<size_t>(out_info.height)
                          * num_out_bands * sizeof(float);

    // Dynamic RAM budget: engine_focal/terrain/texture pre-claims 90% of available RAM
    // for its pinned halo pool (init_ram_budget). The output array must fit in the
    // remaining headroom. Plain algebra/clip/reproject pipelines can use most of RAM.
    double ram_fraction;
    if (tmp_ctx.has_focal || tmp_ctx.has_terrain || tmp_ctx.has_texture) {
        // engine_focal claims 0.90 * available_ram → only 0.10 left for output
        // Use 0.08 to stay safely below that ceiling.
        ram_fraction = 0.08;
    } else if (tmp_ctx.has_temporal) {
        // Temporal loads N full-scene buffers simultaneously.
        ram_fraction = 0.25;
    } else {
        // Simple algebra / clip / reproject — very little internal overhead.
        ram_fraction = 0.70;
    }

    size_t safe_ram_limit = static_cast<size_t>(get_available_ram() * ram_fraction);

    if (required_bytes > safe_ram_limit) {
        throw std::runtime_error(
            "MemoryError: raster requires " + std::to_string(required_bytes / 1048576)
            + " MB but only " + std::to_string(safe_ram_limit / 1048576)
            + " MB is safely available for output allocation (pipeline mode: "
            + (tmp_ctx.has_focal || tmp_ctx.has_terrain || tmp_ctx.has_texture
                ? "focal/terrain/texture" : tmp_ctx.has_temporal ? "temporal" : "standard")
            + "). Use iter_begin() to stream chunks instead.");
    }

    auto result      = std::make_shared<RasterResult>();
    result->width    = out_info.width;
    result->height   = out_info.height;
    result->bands    = num_out_bands;
    result->file_info   = out_info;
    result->projection  = out_info.projection;
    memcpy(result->geo_transform, out_info.geo_transform, sizeof(out_info.geo_transform));

    try {
        result->allocate();
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "MemoryError: raster requires " +
            std::to_string(required_bytes / 1048576) +
            " MB but allocation failed. Use iter_begin() to stream chunks instead.");
    }

    execute(nullptr, result.get(), nullptr, verbose);
    return result;
}

std::shared_ptr<ChunkQueue> Chain::iter_begin(int buffer_chunk_count) {
    auto queue = std::make_shared<ChunkQueue>(buffer_chunk_count);
    Chain chain_copy(*this);

    std::thread([chain_copy, queue]() mutable {
        try {
            chain_copy.execute(nullptr, nullptr, queue);
        } catch (...) {
            
        }
        queue->finish();
    }).detach();

    return queue;
}
