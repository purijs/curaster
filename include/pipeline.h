/**
 * @file pipeline.h
 * @brief Pipeline context types connecting the Chain API to the engine.
 *
 * Defines the operation types a Chain can hold (algebra, clip, reproject,
 * reclass), the reprojection parameters struct, and the PipelineCtx that
 * the engine consumes to drive GPU execution.
 */
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "raster_core.h"  // Instruction, GpuSpanRow
#include "types.h"        // FileInfo, ResampleMethod

// Forward declaration to avoid pulling in GDAL headers here.
class GDALRasterBand;

// ─── Reprojection parameters ──────────────────────────────────────────────────
/**
 * @brief Parameters for the REPROJECT chain operation.
 */
struct ReprojectParams {
    std::string   target_crs;            ///< Target CRS as any string GDAL understands (EPSG, WKT, proj)
    double        pixel_size_x  = 0;     ///< Output pixel width  (0 = auto-derive from source)
    double        pixel_size_y  = 0;     ///< Output pixel height (0 = auto-derive from source)
    ResampleMethod resampling   = ResampleMethod::BILINEAR; ///< Resampling algorithm
    double        nodata_value  = -9999.0; ///< Fill value for pixels outside the source extent

    // Optional output extent in target CRS units.
    bool   has_extent  = false;   ///< True if the four extent fields below have been set
    double extent_xmin = 0;
    double extent_ymin = 0;
    double extent_xmax = 0;
    double extent_ymax = 0;
};

// ─── Chain operation types ────────────────────────────────────────────────────
/**
 * @brief Discriminator for the different operations a Chain can hold.
 */
enum class ChainOpType {
    ALGEBRA,    ///< Band-math expression (e.g. "(B1 - B2) / B3")
    RECLASS,    ///< Reclassification (future)
    CLIP,       ///< Polygon clip via GeoJSON
    REPROJECT,  ///< Coordinate reprojection
};

/**
 * @brief A single lazy operation stored inside a Chain.
 *
 * Only one of the union-like fields is meaningful, determined by @c type.
 */
struct ChainOp {
    ChainOpType type;

    // ALGEBRA payload
    std::string algebra_expr;

    // CLIP payload
    std::string geojson_str;

    // RECLASS payload (reserved for future use)
    std::vector<std::pair<double, double>> reclass_src;
    std::vector<double>                    reclass_dst;

    // REPROJECT payload
    ReprojectParams reproject_params;
};

// ─── Pipeline execution context ───────────────────────────────────────────────
/**
 * @brief Fully resolved context consumed by run_engine_ex / run_engine_reproject.
 *
 * Built from a Chain's operation list by chain_to_ctx() in chain.cpp.
 * The context is passed by reference through the entire engine call to avoid
 * redundant copying of instruction arrays and span tables.
 */
struct PipelineCtx {
    // ── Compiled raster algebra program ──────────────────────────────────────
    std::vector<Instruction> instructions; ///< Compiled algebra VM instructions
    std::vector<int>         band_map;     ///< Mapping: virtual band slot → physical band index

    // ── Polygon clip ──────────────────────────────────────────────────────────
    bool has_clip_mask = false; ///< True if this pipeline includes a clip operation
    /// Per-row span arrays derived from the clip GeoJSON, keyed by image row.
    std::unordered_map<int, std::vector<GpuSpanRow>> clip_spans;

    // ── Reclassification (reserved for future use) ────────────────────────────
    std::vector<std::pair<int, float>> reclass_ranges;

    // ── Output callbacks / destinations ──────────────────────────────────────
    /// If set, each completed chunk is written here (GDAL write path).
    GDALRasterBand* output_band = nullptr;

    /// Called with (width, height, host_ptr, y_offset) for each completed chunk.
    std::function<void(int, int, float*, int)> result_callback;

    /// Same signature as result_callback — used by the streaming queue path.
    std::function<void(int, int, float*, int)> queue_callback;

    // ── Reprojection ──────────────────────────────────────────────────────────
    bool             has_reproject       = false;
    ReprojectParams  reproject_params;
    FileInfo         reproject_output_info; ///< Pre-computed output FileInfo for the warp path
};
