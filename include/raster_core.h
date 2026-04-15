/**
 * @file raster_core.h
 * @brief Shared GPU types and kernel launch declarations.
 *
 * Included by both CPU (.cpp) and CUDA (.cu) translation units.
 * Defines the raster-algebra instruction set, polygon-clip span types,
 * and the public launch wrappers for all CUDA kernels.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

// ─── Warp kernel coarse-grid dimensions ──────────────────────────────────────
// The GPU reprojection kernel uses a 32×32 control-point grid stored in
// shared memory to approximate per-pixel coordinate transforms cheaply.
#define WARP_GRID_WIDTH  32
#define WARP_GRID_HEIGHT 32

// ─── Raster algebra virtual-machine opcodes ───────────────────────────────────
/**
 * @brief Opcodes for the stack-based per-pixel raster algebra VM.
 *
 * algebra_compiler.cpp translates expressions like "(B1 - B2) / 2.0"
 * into a sequence of these opcodes that the GPU kernel evaluates
 * independently for every output pixel.
 */
enum Opcode {
    OP_ADD,        ///< Pop b, a → push (a + b)
    OP_SUB,        ///< Pop b, a → push (a - b)
    OP_MUL,        ///< Pop b, a → push (a * b)
    OP_DIV,        ///< Pop b, a → push (a / (b + ε)) — epsilon-safe division
    OP_LOAD_BAND,  ///< Push band[band_index] value at the current pixel
    OP_LOAD_CONST, ///< Push a scalar constant onto the stack
    OP_GT,         ///< Pop b, a → push (a >  b ? 1.0 : 0.0)
    OP_LT,         ///< Pop b, a → push (a <  b ? 1.0 : 0.0)
    OP_GTE,        ///< Pop b, a → push (a >= b ? 1.0 : 0.0)
    OP_LTE,        ///< Pop b, a → push (a <= b ? 1.0 : 0.0)
    OP_EQ,         ///< Pop b, a → push (a == b ? 1.0 : 0.0)
    OP_NEQ,        ///< Pop b, a → push (a != b ? 1.0 : 0.0)
    OP_AND,        ///< Pop b, a → push (a * b)      (fuzzy AND)
    OP_OR,         ///< Pop b, a → push max(a, b)    (fuzzy OR)
    OP_NOT,        ///< Replace top-of-stack → push (1.0 - top)
    OP_IF,         ///< Pop false_val, true_val, cond → push cond·true + (1−cond)·false
    OP_BETWEEN,    ///< Pop hi, lo, x → push (lo <= x && x <= hi ? 1.0 : 0.0)
    OP_CLAMP,      ///< Pop hi, lo, x → push clamp(x, lo, hi)
    OP_MIN2,       ///< Pop b, a → push min(a, b)
    OP_MAX2,       ///< Pop b, a → push max(a, b)
};

// ─── Algebra VM instruction ────────────────────────────────────────────────────
/**
 * @brief One compiled instruction for the raster algebra virtual machine.
 */
struct Instruction {
    Opcode op;         ///< Operation to perform
    float  constant;   ///< Scalar value for OP_LOAD_CONST  (unused by other ops)
    int    band_index; ///< Band slot index for OP_LOAD_BAND (unused by other ops)
};

// ─── Polygon clip span types ──────────────────────────────────────────────────
/**
 * @brief An inclusive column range [start, end] produced by polygon scan-conversion.
 */
struct GpuSpanPair {
    int start; ///< First column inside the polygon (inclusive)
    int end;   ///< Last  column inside the polygon (inclusive)
};

/**
 * @brief All fill-spans for one raster row, stored in a fixed-size POD array.
 *
 * Kept as zero-copy pinned host memory so the GPU can access it directly without
 * an explicit DMA transfer. Supports up to 1 024 spans per row.
 */
struct GpuSpanRow {
    int         num_spans;     ///< Number of valid entries in spans[]
    GpuSpanPair spans[1024];  ///< Horizontal column ranges covering this row
};

// ─── CUDA kernel launch declarations (implemented in src/gpu/) ────────────────

/**
 * @brief Execute the compiled raster algebra program over @p num_pixels pixels.
 *
 * @param d_instructions   Device pointer to the compiled Instruction array.
 * @param num_instructions Number of instructions in the program.
 * @param d_bands          Device pointer-array: one entry per virtual band slot.
 * @param d_output         Device output buffer (one float per pixel).
 * @param num_pixels       Total number of pixels to process.
 * @param stream           CUDA stream for asynchronous execution.
 */
void launch_raster_algebra(
    const Instruction*  d_instructions,
    int                 num_instructions,
    const float* const* d_bands,
    float*              d_output,
    size_t              num_pixels,
    cudaStream_t        stream);

/**
 * @brief Zero-out every pixel that falls outside all polygon spans (in-place).
 *
 * @param d_output    Buffer to mask in-place.
 * @param num_pixels  Total pixels (image_width × chunk_height).
 * @param image_width Used to map a flat pixel index to (row, col).
 * @param d_spans     Per-row span table (pinned, GPU-accessible).
 * @param stream      CUDA stream.
 */
void launch_apply_mask(
    float*             d_output,
    size_t             num_pixels,
    int                image_width,
    const GpuSpanRow*  d_spans,
    cudaStream_t       stream);

/**
 * @brief Sample source textures through the coarse warp grid and write to output.
 *
 * @param d_coarse_x        Control-point source X coordinates (device).
 * @param d_coarse_y        Control-point source Y coordinates (device).
 * @param d_textures        CUDA texture object handles, one per band.
 * @param d_output_flat     Output: [band * max_chunk_pixels + (y * dst_width + x)].
 * @param canvas_width      Width of the source canvas in pixels.
 * @param canvas_height     Height of the source canvas in pixels.
 * @param dst_width         Destination tile width in pixels.
 * @param dst_height        Destination tile height in pixels.
 * @param num_bands         Number of spectral bands.
 * @param nodata_value      Written to pixels that project outside the canvas.
 * @param max_chunk_pixels  Stride between band planes in d_output_flat.
 * @param stream            CUDA stream.
 */
void launch_warp_kernel(
    const float*               d_coarse_x,
    const float*               d_coarse_y,
    const cudaTextureObject_t* d_textures,
    float*                     d_output_flat,
    int                        canvas_width,
    int                        canvas_height,
    int                        dst_width,
    int                        dst_height,
    int                        num_bands,
    float                      nodata_value,
    size_t                     max_chunk_pixels,
    cudaStream_t               stream);

// ─── Focal / neighborhood kernel ─────────────────────────────────────────────

void launch_focal_kernel(
    const float*  d_halo_src,
    float*        d_output,
    int           src_width,
    int           halo_height,
    int           dst_width,
    int           dst_height,
    int           radius,
    int           stat_id,
    int           shape_circle,   // 0=square 1=circle
    cudaStream_t  stream);

// ─── Terrain kernel ───────────────────────────────────────────────────────────

void launch_terrain_kernel(
    const float*  d_halo_src,
    float*        d_output,
    int           src_width,
    int           halo_height,
    int           dst_width,
    int           dst_height,
    uint32_t      features_mask,
    float         cell_x,
    float         cell_y,
    float         sun_az_rad,
    float         sun_alt_rad,
    bool          use_zevenbergen,
    int           unit_mode,       // 0=degrees 1=radians 2=percent
    size_t        max_chunk_pixels,
    cudaStream_t  stream);

// ─── GLCM kernel ──────────────────────────────────────────────────────────────

void launch_glcm_kernel(
    const float*  d_halo_src,
    float*        d_output,
    int           src_width,
    int           halo_height,
    int           dst_width,
    int           dst_height,
    int           window,
    int           levels,
    float         val_min,
    float         val_max,
    int           dx,
    int           dy,
    int           num_output_features,
    size_t        max_chunk_pixels,
    bool          log_scale,
    cudaStream_t  stream);

void launch_glcm_avg_divide(
    float*        d_output,
    int           num_pixels,
    int           num_features,
    int           num_dirs,
    cudaStream_t  stream);

// ─── Zonal reduction kernel ───────────────────────────────────────────────────

void launch_zonal_reduction(
    const float*    d_values,
    const uint16_t* d_zone_labels,
    size_t          num_pixels,
    int*            d_count,
    float*          d_sum,
    float*          d_sum_sq,
    float*          d_min,
    float*          d_max,
    int             num_zones,
    cudaStream_t    stream);

// ─── Temporal stack kernel ────────────────────────────────────────────────────

void launch_temporal_kernel(
    const float** d_scene_ptrs,
    float*        d_output,
    size_t        num_pixels,
    int           num_scenes,
    int           op_id,
    int           t0_idx,
    int           t1_idx,
    const float*  d_time_values,
    float         denominator,
    cudaStream_t  stream);
