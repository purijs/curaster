/**
 * @file engine.h
 * @brief Declarations for the curaster processing engines and RAM budget helpers.
 *
 * Two engines are provided:
 *  - run_engine_direct:    Direct read path (no reprojection).
 *  - run_engine_reproject: Warp path (with reprojection).
 *
 * The correct engine is selected automatically inside run_engine_ex()
 * based on whether ctx.has_reproject is set.
 */
#pragma once

#include <cstddef>
#include "../../include/pipeline.h"  // PipelineCtx

// ─── RAM budget ───────────────────────────────────────────────────────────────

/// Maximum number of OMP threads used for parallel chunk processing.
extern int    g_num_threads;

/// Pinned-memory budget in bytes (set to ~55% of available RAM at startup).
extern size_t g_pinned_budget;

/**
 * @brief Initialise the pinned-memory RAM budget from available system RAM.
 *
 * Safe to call multiple times — no-op after the first call.
 */
void init_ram_budget();

/// Return the number of bytes of physical RAM currently available to the process.
size_t get_available_ram();

// ─── Engine entry point ───────────────────────────────────────────────────────

/**
 * @brief Execute the processing pipeline described by @p ctx on @p input_file.
 *
 * Dispatches to run_engine_reproject() if @p ctx.has_reproject is true,
 * otherwise runs the direct (no-warp) engine.
 *
 * Results are delivered via ctx.output_band, ctx.result_callback, and/or
 * ctx.queue_callback, whichever are non-null.
 *
 * @param input_file  Path to the source GeoTIFF (local or S3 URI).
 * @param ctx         Fully populated pipeline context.
 * @param verbose     If true, print a GDAL-style progress bar to stdout.
 */
void run_engine_ex(const std::string& input_file, PipelineCtx& ctx, bool verbose);

/**
 * @brief Warp-path engine: reproject the source, then apply algebra + clip.
 *
 * Called by run_engine_ex() when ctx.has_reproject is true. Reads source
 * tiles into a per-thread canvas, runs the GPU warp kernel, then continues
 * with the same algebra and clip logic as the direct path.
 */
void run_engine_reproject(const std::string& input_file, PipelineCtx& ctx, bool verbose);
