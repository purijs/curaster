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




#define WARP_GRID_WIDTH  32
#define WARP_GRID_HEIGHT 32


/**
 * @brief Opcodes for the stack-based per-pixel raster algebra VM.
 *
 * algebra_compiler.cpp translates expressions like "(B1 - B2) / 2.0"
 * into a sequence of these opcodes that the GPU kernel evaluates
 * independently for every output pixel.
 */
enum Opcode {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_LOAD_BAND,
    OP_LOAD_CONST,
    OP_GT,
    OP_LT,
    OP_GTE,
    OP_LTE,
    OP_EQ,
    OP_NEQ,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_IF,
    OP_BETWEEN,
    OP_CLAMP,
    OP_MIN2,
    OP_MAX2,
};


/**
 * @brief One compiled instruction for the raster algebra virtual machine.
 */
struct Instruction {
    Opcode op;
    float  constant;
    int    band_index;
};


/**
 * @brief An inclusive column range [start, end] produced by polygon scan-conversion.
 */
struct GpuSpanPair {
    int start;
    int end;
};

/**
 * @brief All fill-spans for one raster row, stored in a fixed-size POD array.
 *
 * Kept as zero-copy pinned host memory so the GPU can access it directly without
 * an explicit DMA transfer. Supports up to 1 024 spans per row.
 */
struct GpuSpanRow {
    int         num_spans;
    GpuSpanPair spans[1024];
};



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



void launch_focal_kernel(
    const float*  d_halo_src,
    float*        d_output,
    int           src_width,
    int           halo_height,
    int           dst_width,
    int           dst_height,
    int           radius,
    int           stat_id,
    int           shape_circle,   
    cudaStream_t  stream);



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
    int           unit_mode,       
    size_t        max_chunk_pixels,
    cudaStream_t  stream);



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
