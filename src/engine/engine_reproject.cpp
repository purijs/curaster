/**
 * @file engine_reproject.cpp
 * @brief Warp-path engine: per-chunk reprojection via GPU texture sampling.
 */
#include "engine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "cpl_progress.h"
#include "gdal_priv.h"

#include "../../include/cuda_utils.h"
#include "../../include/raster_core.h"
#include "../../include/thread_buffers.h"

#include "../reproject/reproject.h"
#include "../s3/s3_auth.h"
#include "../s3/s3_fetch.h"
#include "../tiff/tiff_metadata.h"
#include "../tile_io/tile_io.h"

// ─── Local GDAL dataset pool
// ──────────────────────────────────────────────────

struct DatasetPool {
  std::vector<GDALDataset *> datasets;
  std::vector<std::vector<GDALRasterBand *>> bands;

  DatasetPool(int n, int nb)
      : datasets(n, nullptr),
        bands(n, std::vector<GDALRasterBand *>(nb, nullptr)) {}

  ~DatasetPool() {
    for (auto *ds : datasets) {
      if (ds) {
        GDALClose(ds);
      }
    }
  }
};

// ─── Persistent warp thread-buffer pool ──────────────────────────────────────
//
// alloc_warp allocates CUDA arrays and texture objects — expensive on first
// call. Persistent pool reuses them across repeated calls with the same raster
// shape.

struct WarpPoolKey {
  size_t output_band_bytes;
  int num_bands;
  int num_threads;
  bool has_clip;
  int max_canvas_width;
  int max_canvas_height;
  size_t max_dst_pixels;
  int resample_method;
  bool operator==(const WarpPoolKey &o) const noexcept {
    return output_band_bytes == o.output_band_bytes &&
           num_bands == o.num_bands && num_threads == o.num_threads &&
           has_clip == o.has_clip && max_canvas_width == o.max_canvas_width &&
           max_canvas_height == o.max_canvas_height &&
           max_dst_pixels == o.max_dst_pixels &&
           resample_method == o.resample_method;
  }
};

static std::vector<ThreadBufs> g_warp_pool;
static WarpPoolKey g_warp_key{0, 0, 0, false, 0, 0, 0, 0};

static std::vector<ThreadBufs> &get_persistent_warp_pool(
    size_t output_band_bytes, int num_bands, int num_threads, bool has_clip,
    int clip_rows, int max_canvas_width, int max_canvas_height,
    size_t max_dst_pixels, ResampleMethod resample_method) {
  WarpPoolKey needed{output_band_bytes, num_bands,
                     num_threads,       has_clip,
                     max_canvas_width,  max_canvas_height,
                     max_dst_pixels,    static_cast<int>(resample_method)};
  if (needed == g_warp_key && !g_warp_pool.empty()) {
    return g_warp_pool;
  }
  for (auto &buf : g_warp_pool) {
    buf.free_all();
  }
  g_warp_pool.clear();
  g_warp_pool.resize(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    g_warp_pool[i].alloc(output_band_bytes, num_bands,
                         has_clip ? clip_rows : 0);
    g_warp_pool[i].alloc_warp(max_canvas_width, max_canvas_height,
                              max_dst_pixels, num_bands, resample_method);
  }
  g_warp_key = needed;
  return g_warp_pool;
}

// ─── run_engine_reproject ────────────────────────────────────────────────────

void run_engine_reproject(const std::string &input_file, PipelineCtx &ctx,
                          bool verbose) {
  GDALAllRegister();
  init_ram_budget();

  FileInfo src_info = get_file_info(input_file);
  const FileInfo &out_info = ctx.reproject_output_info;
  const bool is_pixel_interleaved = (src_info.interleave == "PIXEL");
  const bool is_s3 = is_s3_path(input_file);
  S3Loc s3_location;
  if (is_s3) {
    s3_location = parse_s3_path(input_file);
  }

  const std::vector<int> &band_slots = ctx.band_map;
  const int num_bands = static_cast<int>(band_slots.size());

  const int chunk_height = out_info.tile_height;
  const int num_chunks = (out_info.height + chunk_height - 1) / chunk_height;

  // ── Estimate input/output pixel scale ratio ────────────────────────────────

  double h_scale = 1.0, v_scale = 1.0;
  {
    WarpTransformer probe;
    probe.initialise(src_info, out_info);

    double probe_x[3] = {static_cast<double>(out_info.width) / 2,
                         static_cast<double>(out_info.width) / 2 + 100,
                         static_cast<double>(out_info.width) / 2};
    double probe_y[3] = {static_cast<double>(out_info.height) / 2,
                         static_cast<double>(out_info.height) / 2,
                         static_cast<double>(out_info.height) / 2 + 100};

    probe.transform_pixels(probe_x, probe_y, 3);

    double dx = std::sqrt(std::pow(probe_x[1] - probe_x[0], 2) +
                          std::pow(probe_y[1] - probe_y[0], 2)) /
                100.0;
    double dy = std::sqrt(std::pow(probe_x[2] - probe_x[0], 2) +
                          std::pow(probe_y[2] - probe_y[0], 2)) /
                100.0;

    if (dx > 0 && dx < 1000) {
      h_scale = dx;
    }
    if (dy > 0 && dy < 1000) {
      v_scale = dy;
    }
    probe.destroy();
  }

  // ── Canvas size bounds ─────────────────────────────────────────────────────

  int max_canvas_width =
      std::min(src_info.width, static_cast<int>(out_info.width * h_scale * 1.5 +
                                                src_info.tile_width * 2 + 128));
  int max_canvas_height = std::min(
      src_info.height, static_cast<int>(chunk_height * v_scale * 1.5 +
                                        src_info.tile_height * 2 + 128));

  size_t canvas_bytes_per_thread = static_cast<size_t>(max_canvas_width) *
                                   max_canvas_height * num_bands *
                                   sizeof(float);
  size_t output_band_bytes =
      static_cast<size_t>(out_info.width) * chunk_height * sizeof(float);
  size_t max_dst_pixels = static_cast<size_t>(out_info.width) * chunk_height;

  // ── Scale thread count to fit within VRAM and pinned RAM ──────────────────

  size_t free_vram = 0, total_vram = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  size_t vram_budget = static_cast<size_t>(free_vram * 0.90);

  int warp_threads = g_num_threads;
  if (canvas_bytes_per_thread > 0) {
    warp_threads = std::min(
        warp_threads, static_cast<int>(std::max(
                          size_t{1}, vram_budget / canvas_bytes_per_thread)));
  }
  size_t pinned_per_thread =
      output_band_bytes * num_bands + canvas_bytes_per_thread;
  if (pinned_per_thread > 0) {
    warp_threads =
        std::min(warp_threads,
                 static_cast<int>(
                     std::max(size_t{1}, g_pinned_budget / pinned_per_thread)));
  }

  if (verbose) {
    printf("[Reproject] Using %d threads within %zu MB VRAM budget.\n",
           warp_threads, vram_budget / (1024 * 1024));
    fflush(stdout);
  }

  // ── Persistent warp thread pool ────────────────────────────────────────────

  auto &thread_pool = get_persistent_warp_pool(
      output_band_bytes, num_bands, warp_threads, ctx.has_clip_mask,
      chunk_height, max_canvas_width, max_canvas_height, max_dst_pixels,
      ctx.reproject_params.resampling);

  // ── Per-call WarpTransformers (cheap to init, not worth caching) ──────────

  std::vector<WarpTransformer> transformers(warp_threads);
  for (int t = 0; t < warp_threads; ++t) {
    transformers[t].initialise(src_info, out_info);
  }

  // ── Open GDAL datasets (non-S3) ───────────────────────────────────────────

  std::vector<int> gdal_band_nums(num_bands);
  for (int b = 0; b < num_bands; ++b) {
    gdal_band_nums[b] = band_slots[b] + 1;
  }

  DatasetPool dataset_pool(warp_threads, num_bands);
  if (!is_s3) {
    for (int t = 0; t < warp_threads; ++t) {
      dataset_pool.datasets[t] =
          static_cast<GDALDataset *>(GDALOpen(input_file.c_str(), GA_ReadOnly));
      if (!dataset_pool.datasets[t]) {
        throw std::runtime_error("Open failed: " + input_file);
      }
      for (int b = 0; b < num_bands; ++b) {
        dataset_pool.bands[t][b] =
            dataset_pool.datasets[t]->GetRasterBand(band_slots[b] + 1);
      }
    }
  }

  // ── Upload compiled algebra program to the GPU ─────────────────────────────

  Instruction *d_instructions = nullptr;
  if (!ctx.instructions.empty()) {
    CUDA_CHECK(cudaMalloc(&d_instructions,
                          ctx.instructions.size() * sizeof(Instruction)));
    CUDA_CHECK(cudaMemcpy(d_instructions, ctx.instructions.data(),
                          ctx.instructions.size() * sizeof(Instruction),
                          cudaMemcpyHostToDevice));
  }

  size_t tiles_across =
      ((size_t)src_info.width + src_info.tile_width - 1) / src_info.tile_width;
  size_t tiles_down = ((size_t)src_info.height + src_info.tile_height - 1) /
                      src_info.tile_height;
  size_t coarse_grid_bytes = WARP_GRID_WIDTH * WARP_GRID_HEIGHT * sizeof(float);

  std::atomic<int> completed_chunks{0};
  double progress_state = -1.0;
  if (verbose) {
    GDALTermProgress(0.0, nullptr, &progress_state);
  }

  // ── Parallel chunk processing ──────────────────────────────────────────────

  omp_set_num_threads(warp_threads);
#pragma omp parallel for schedule(dynamic, 1)
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int thread_id = omp_get_thread_num();
    int chunk_y0 = chunk * chunk_height;
    int cur_height = std::min(chunk_height, out_info.height - chunk_y0);
    ThreadBufs &buf = thread_pool[thread_id];
    WarpTransformer &warp = transformers[thread_id];

    // ── Phase 1: Build coarse grid ─────────────────────────────────────────

    double coarse_grid_x[WARP_GRID_WIDTH * WARP_GRID_HEIGHT];
    double coarse_grid_y[WARP_GRID_WIDTH * WARP_GRID_HEIGHT];
    compute_coarse_grid(warp, chunk_y0, cur_height, out_info.width,
                        coarse_grid_x, coarse_grid_y);

    // ── Phase 2: Derive source canvas bounding box ────────────────────────

    SrcBBox canvas_bbox = coarse_grid_to_source_bbox(
        coarse_grid_x, coarse_grid_y, src_info.width, src_info.height);
    canvas_bbox.w = std::min(canvas_bbox.w, max_canvas_width);
    canvas_bbox.h = std::min(canvas_bbox.h, max_canvas_height);

    float coarse_x_f[WARP_GRID_WIDTH * WARP_GRID_HEIGHT];
    float coarse_y_f[WARP_GRID_WIDTH * WARP_GRID_HEIGHT];
    for (int i = 0; i < WARP_GRID_WIDTH * WARP_GRID_HEIGHT; ++i) {
      coarse_x_f[i] = static_cast<float>(coarse_grid_x[i] - canvas_bbox.x0);
      coarse_y_f[i] = static_cast<float>(coarse_grid_y[i] - canvas_bbox.y0);
    }

    // ── Phase 3: Read source canvas from GDAL or S3 ───────────────────────

    memset(buf.h_src_canvas, 0,
           static_cast<size_t>(canvas_bbox.w) * canvas_bbox.h * num_bands *
               sizeof(float));

    if (!is_s3) {
      for (int b = 0; b < num_bands; ++b) {
        dataset_pool.bands[thread_id][b]->RasterIO(
            GF_Read, canvas_bbox.x0, canvas_bbox.y0, canvas_bbox.w,
            canvas_bbox.h, buf.h_src_bands[b], canvas_bbox.w, canvas_bbox.h,
            GDT_Float32, 0, 0);
      }

    } else if (src_info.is_tiled) {
      std::vector<TileFetch> fetch_jobs;
      int tr0 = canvas_bbox.y0 / src_info.tile_height;
      int tr1 = (canvas_bbox.y0 + canvas_bbox.h - 1) / src_info.tile_height;
      int tc0 = canvas_bbox.x0 / src_info.tile_width;
      int tc1 = (canvas_bbox.x0 + canvas_bbox.w - 1) / src_info.tile_width;

      for (int tr = tr0; tr <= tr1; ++tr) {
        for (int tc = tc0; tc <= tc1; ++tc) {
          if (is_pixel_interleaved) {
            size_t ti = static_cast<size_t>(tr) * tiles_across + tc;
            if (ti < src_info.tile_offsets.size()) {
              fetch_jobs.push_back({ti,
                                    src_info.tile_offsets[ti],
                                    src_info.tile_lengths[ti],
                                    {},
                                    0});
            }
          } else {
            for (int b = 0; b < num_bands; ++b) {
              size_t ti = static_cast<size_t>(band_slots[b]) * tiles_across *
                              tiles_down +
                          static_cast<size_t>(tr) * tiles_across + tc;
              if (ti < src_info.tile_offsets.size()) {
                fetch_jobs.push_back({ti,
                                      src_info.tile_offsets[ti],
                                      src_info.tile_lengths[ti],
                                      {},
                                      0});
              }
            }
          }
        }
      }

      s3_fetch_tiles(s3_location, src_info.tile_offsets, src_info.tile_lengths,
                     fetch_jobs);

      for (auto &job : fetch_jobs) {
        if (job.err || job.data.empty()) {
          continue;
        }
        int block_spp = is_pixel_interleaved ? src_info.samples_per_pixel : 1;
        auto float_tile = decompress_tile_to_float(
            job.data.data(), job.data.size(), src_info.tile_width,
            src_info.tile_height, block_spp, src_info.compression,
            src_info.predictor, src_info.data_type);

        size_t ti = job.tile_index;
        int tr = static_cast<int>(ti / tiles_across);
        int tc = static_cast<int>(ti % tiles_across);
        int band_plane = -1;
        if (!is_pixel_interleaved) {
          int b = static_cast<int>(ti / (tiles_across * tiles_down));
          tr = static_cast<int>((ti % (tiles_across * tiles_down)) /
                                tiles_across);
          tc = static_cast<int>(ti % tiles_across);
          band_plane = b;
        }

        fill_canvas_from_tile(float_tile.data(), tr * src_info.tile_height,
                              tc * src_info.tile_width, src_info.tile_width,
                              src_info.tile_height, block_spp,
                              is_pixel_interleaved, band_slots, buf.h_src_bands,
                              canvas_bbox, band_plane);
      }

    } else {
      std::vector<TileFetch> fetch_jobs;
      int strips_per_band = (src_info.height + src_info.rows_per_strip - 1) /
                            src_info.rows_per_strip;
      int sr0 = canvas_bbox.y0 / src_info.rows_per_strip;
      int sr1 = (canvas_bbox.y0 + canvas_bbox.h - 1) / src_info.rows_per_strip;

      for (size_t slot = 0; slot < band_slots.size(); ++slot) {
        int physical_band = band_slots[slot];
        for (int sr = sr0; sr <= sr1; ++sr) {
          size_t si =
              is_pixel_interleaved
                  ? static_cast<size_t>(sr)
                  : static_cast<size_t>(physical_band) * strips_per_band + sr;
          if (si < src_info.strip_offsets.size()) {
            fetch_jobs.push_back({si,
                                  src_info.strip_offsets[si],
                                  src_info.strip_lengths[si],
                                  {},
                                  0});
          }
        }
        if (is_pixel_interleaved) {
          break;
        }
      }

      s3_fetch_tiles(s3_location, src_info.strip_offsets,
                     src_info.strip_lengths, fetch_jobs);

      for (auto &job : fetch_jobs) {
        if (job.err || job.data.empty()) {
          continue;
        }
        int sr = static_cast<int>(job.tile_index);
        if (!is_pixel_interleaved) {
          sr = sr % strips_per_band;
        }

        int strip_first_row = sr * src_info.rows_per_strip;
        int strip_num_rows = std::min(src_info.rows_per_strip,
                                      src_info.height - strip_first_row);
        int block_spp = is_pixel_interleaved ? src_info.samples_per_pixel : 1;

        auto float_strip = decompress_tile_to_float(
            job.data.data(), job.data.size(), src_info.width, strip_num_rows,
            block_spp, src_info.compression, src_info.predictor,
            src_info.data_type);

        fill_canvas_from_strip(float_strip.data(), strip_first_row,
                               strip_num_rows, src_info.width, block_spp,
                               is_pixel_interleaved, band_slots,
                               buf.h_src_bands, canvas_bbox, -1);
      }
    }

    // ── Phase 4: Upload coarse grid and canvas textures to GPU ────────────

    CUDA_CHECK(cudaMemcpyAsync(buf.d_coarse_x, coarse_x_f, coarse_grid_bytes,
                               cudaMemcpyHostToDevice, buf.cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_coarse_y, coarse_y_f, coarse_grid_bytes,
                               cudaMemcpyHostToDevice, buf.cuda_stream));
    buf.bind_src_canvas(num_bands, canvas_bbox.w, canvas_bbox.h,
                        buf.cuda_stream);

    // ── Phase 5: Launch warp kernel ───────────────────────────────────────

    launch_warp_kernel(
        buf.d_coarse_x, buf.d_coarse_y,
        reinterpret_cast<const cudaTextureObject_t *>(buf.d_texture_objects),
        buf.d_warp_output, canvas_bbox.w, canvas_bbox.h, out_info.width,
        cur_height, num_bands,
        static_cast<float>(ctx.reproject_params.nodata_value),
        static_cast<size_t>(out_info.width) * chunk_height, buf.cuda_stream);

    // ── Phase 6: Launch algebra kernel ────────────────────────────────────

    size_t dst_pixels = static_cast<size_t>(out_info.width) * cur_height;
    launch_raster_algebra(
        d_instructions, static_cast<int>(ctx.instructions.size()),
        const_cast<const float *const *>(buf.d_warp_band_ptrs), buf.d_output,
        dst_pixels, buf.cuda_stream);

    // ── Phase 7: Launch mask kernel (if clip is active) ───────────────────

    if (ctx.has_clip_mask && buf.d_clip_spans) {
      memset(buf.h_clip_spans, 0, cur_height * sizeof(GpuSpanRow));
      for (int row = chunk_y0; row < chunk_y0 + cur_height; ++row) {
        auto it = ctx.clip_spans.find(row);
        if (it != ctx.clip_spans.end() && !it->second.empty()) {
          buf.h_clip_spans[row - chunk_y0] = it->second[0];
        }
      }
      launch_apply_mask(buf.d_output, dst_pixels, out_info.width,
                        buf.d_clip_spans, buf.cuda_stream);
    }

    // ── Phase 8: Synchronise and deliver results ───────────────────────────

    CUDA_CHECK(cudaStreamSynchronize(buf.cuda_stream));

    if (ctx.output_band) {
#pragma omp critical(gdal_write)
      {
        ctx.output_band->RasterIO(GF_Write, 0, chunk_y0, out_info.width,
                                  cur_height, buf.h_output, out_info.width,
                                  cur_height, GDT_Float32, 0, 0);
      }
    }
    if (ctx.queue_callback) {
      ctx.queue_callback(out_info.width, cur_height, buf.h_output, chunk_y0);
    }
    if (ctx.result_callback) {
      ctx.result_callback(out_info.width, cur_height, buf.h_output, chunk_y0);
    }

    if (verbose) {
      int done = ++completed_chunks;
#pragma omp critical
      GDALTermProgress(static_cast<double>(done) / num_chunks, nullptr,
                       &progress_state);
    }
  }

  if (d_instructions) {
    cudaFree(d_instructions);
  }
  for (auto &warp : transformers) {
    warp.destroy();
  }
  if (verbose) {
    GDALTermProgress(1.0, nullptr, &progress_state);
  }
}