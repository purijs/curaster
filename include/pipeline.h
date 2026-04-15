/**
 * @file pipeline.h
 * @brief Pipeline context types connecting the Chain API to the engine.
 *
 * Defines the operation types a Chain can hold (algebra, clip, reproject,
 * reclass, focal, terrain, texture, zonal_stats), parameter structs, and
 * the PipelineCtx that the engine consumes to drive GPU execution.
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
    ALGEBRA,
    RECLASS,
    CLIP,
    REPROJECT,
    FOCAL,
    TERRAIN,
    TEXTURE,
    ZONAL_STATS,
    TEMPORAL,
};

// ─── Focal statistics parameters ─────────────────────────────────────────────
enum class FocalStat {
    MEAN, SUM, MIN, MAX, STD, VARIANCE, MEDIAN, RANGE
};

enum class FocalShape {
    SQUARE, CIRCLE
};

struct FocalParams {
    FocalStat  stat         = FocalStat::MEAN;
    int        radius       = 1;
    FocalShape shape        = FocalShape::SQUARE;
    bool       clamp_border = true;
};

// ─── Terrain analysis parameters ─────────────────────────────────────────────

#define TERRAIN_SLOPE        (1u << 0)
#define TERRAIN_ASPECT       (1u << 1)
#define TERRAIN_HILLSHADE    (1u << 2)
#define TERRAIN_TRI          (1u << 3)
#define TERRAIN_TPI          (1u << 4)
#define TERRAIN_ROUGHNESS    (1u << 5)
#define TERRAIN_PROF_CURV    (1u << 6)
#define TERRAIN_PLAN_CURV    (1u << 7)
#define TERRAIN_TOTAL_CURV   (1u << 8)

struct TerrainParams {
    std::vector<std::string> metrics;
    std::string unit         = "degrees";
    double      sun_azimuth  = 315.0;
    double      sun_altitude = 45.0;
    std::string method       = "horn";
    float       cell_size_x  = 1.0f;
    float       cell_size_y  = 1.0f;
    uint32_t    features_mask     = 0;
    int         num_output_bands  = 1;
};

// ─── GLCM texture parameters ──────────────────────────────────────────────────
struct GLCMParams {
    std::vector<std::string> features;
    int   window           = 11;
    int   levels           = 32;
    bool  avg_directions   = true;   // true = average 4 directions → 18 bands
    bool  log_scale        = false;  // true = 10*log10(v) before quantize (SAR)
    float value_min        = 0.0f;
    float value_max        = 0.0f;   // value_min==value_max==0 → triggers auto-range
    bool  auto_range       = true;
    int   num_output_bands = 18;     // 18 if avg, 72 if separate
};

// ─── Zonal statistics parameters ─────────────────────────────────────────────
struct ZonalParams {
    std::string              geojson_str;
    std::vector<std::string> stats;
    std::vector<float>       percentiles;
    int                      band = 1;
};

// ─── Zonal statistics result ──────────────────────────────────────────────────
struct ZoneResult {
    int    zone_id = 0;
    double mean    = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double std_dev = 0.0;
    long   count   = 0;
    double sum     = 0.0;
};

// ─── Temporal operation parameters ────────────────────────────────────────────
enum class TemporalOp {
    DIFF, RATIO, ANOMALY_MEAN, ANOMALY_BASELINE, TREND,
    TMEAN, TSTD, TMIN, TMAX
};

struct TemporalParams {
    TemporalOp        op           = TemporalOp::DIFF;
    int               t0_idx       = 0;
    int               t1_idx       = -1;   // -1 = last scene
    int               baseline_idx = -1;   // for ANOMALY_BASELINE
    std::vector<float> time_values;         // scene timestamps (optional)
    float             denominator  = 1.f;  // pre-computed OLS denom for TREND
};

// ─── ChainOp ──────────────────────────────────────────────────────────────────
struct ChainOp {
    ChainOpType type;

    std::string algebra_expr;
    std::string geojson_str;

    std::vector<std::pair<double, double>> reclass_src;
    std::vector<double>                    reclass_dst;

    ReprojectParams reproject_params;
    FocalParams     focal_params;
    TerrainParams   terrain_params;
    GLCMParams      glcm_params;
    ZonalParams     zonal_params;
    TemporalParams  temporal_params;
};

// ─── Pipeline execution context ───────────────────────────────────────────────
struct PipelineCtx {
    std::vector<Instruction> instructions;
    std::vector<int>         band_map;

    bool has_clip_mask = false;
    std::unordered_map<int, std::vector<GpuSpanRow>> clip_spans;

    std::vector<std::pair<int, float>> reclass_ranges;

    GDALRasterBand* output_band = nullptr;
    std::function<void(int, int, float*, int)> result_callback;
    std::function<void(int, int, float*, int)> queue_callback;

    // ── Reprojection ──────────────────────────────────────────────────────────
    bool             has_reproject       = false;
    ReprojectParams  reproject_params;
    FileInfo         reproject_output_info;

    // ── Focal statistics ──────────────────────────────────────────────────────
    bool        has_focal   = false;
    FocalParams focal_params;
    int         focal_num_output_bands = 1;

    // ── Terrain analysis ──────────────────────────────────────────────────────
    bool          has_terrain    = false;
    TerrainParams terrain_params;

    // ── GLCM texture ──────────────────────────────────────────────────────────
    bool       has_texture   = false;
    GLCMParams glcm_params;

    // ── Zonal statistics (terminal) ───────────────────────────────────────────
    bool        has_zonal    = false;
    ZonalParams zonal_params;
    std::vector<ZoneResult> zonal_results;

    // ── Temporal stack (set by StackChain, consumed by engine_temporal) ───────
    bool           has_temporal    = false;
    TemporalParams temporal_params;
    int            temporal_num_scenes = 0;
    std::vector<std::string> temporal_scene_files;
};

