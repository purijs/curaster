/**
 * @file gpu_warp.cu
 * @brief GPU kernel for coordinate-warped texture sampling (reprojection).
 *
 * Each output pixel's source coordinate is computed by bilinear interpolation
 * of a 32×32 coarse control-point grid stored in shared memory.  Storing the
 * grid in shared memory avoids per-pixel global memory reads of the full warp
 * field, which would be prohibitively slow.
 *
 * The kernel writes one float per (band, dest_pixel) pair into d_output_flat,
 * laid out as [band * max_chunk_pixels + (y * dst_width + x)].
 */
#include "../../include/raster_core.h"
#include <cuda_runtime.h>

__global__ void kernel_warp(
    const float* __restrict__           d_coarse_x,
    const float* __restrict__           d_coarse_y,
    const cudaTextureObject_t* __restrict__ d_textures,
    float* __restrict__                 d_output,
    int                                 canvas_width,
    int                                 canvas_height,
    int                                 dst_width,
    int                                 dst_height,
    int                                 num_bands,
    float                               nodata_value,
    size_t                              max_chunk_pixels) {

    
    
    
    
    extern __shared__ float smem[];
    float* shared_coarse_x = smem;
    float* shared_coarse_y = smem + WARP_GRID_WIDTH * WARP_GRID_HEIGHT;

    
    
    
    const int grid_size  = WARP_GRID_WIDTH * WARP_GRID_HEIGHT;
    const int flat_id    = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    for (int i = flat_id; i < grid_size; i += block_size) {
        shared_coarse_x[i] = d_coarse_x[i];
        shared_coarse_y[i] = d_coarse_y[i];
    }
    __syncthreads();

    
    const int dst_col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int dst_row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (dst_col >= dst_width || dst_row >= dst_height) { return; }

    
    const float grid_col_frac = (dst_width  > 1)
        ? static_cast<float>(dst_col) / static_cast<float>(dst_width  - 1)
          * static_cast<float>(WARP_GRID_WIDTH  - 1)
        : 0.f;

    const float grid_row_frac = (dst_height > 1)
        ? static_cast<float>(dst_row) / static_cast<float>(dst_height - 1)
          * static_cast<float>(WARP_GRID_HEIGHT - 1)
        : 0.f;

    
    const int cell_col = min(static_cast<int>(grid_col_frac), WARP_GRID_WIDTH  - 2);
    const int cell_row = min(static_cast<int>(grid_row_frac), WARP_GRID_HEIGHT - 2);

    
    const float frac_x = grid_col_frac - static_cast<float>(cell_col);
    const float frac_y = grid_row_frac - static_cast<float>(cell_row);

    
    const float w00 = (1.f - frac_x) * (1.f - frac_y);
    const float w10 =        frac_x  * (1.f - frac_y);
    const float w01 = (1.f - frac_x) *        frac_y;
    const float w11 =        frac_x  *         frac_y;

    
    const int idx00 =  cell_row      * WARP_GRID_WIDTH + cell_col;
    const int idx10 =  cell_row      * WARP_GRID_WIDTH + cell_col + 1;
    const int idx01 = (cell_row + 1) * WARP_GRID_WIDTH + cell_col;
    const int idx11 = (cell_row + 1) * WARP_GRID_WIDTH + cell_col + 1;

    
    const float src_x =
        w00 * shared_coarse_x[idx00] + w10 * shared_coarse_x[idx10]
      + w01 * shared_coarse_x[idx01] + w11 * shared_coarse_x[idx11];

    const float src_y =
        w00 * shared_coarse_y[idx00] + w10 * shared_coarse_y[idx10]
      + w01 * shared_coarse_y[idx01] + w11 * shared_coarse_y[idx11];

    
    
    const bool outside_canvas =
        (src_x < 0.f || src_y < 0.f
      || src_x >= static_cast<float>(canvas_width)
      || src_y >= static_cast<float>(canvas_height));

    
    const size_t dest_pixel_offset = static_cast<size_t>(dst_row) * dst_width + dst_col;

    for (int band = 0; band < num_bands; ++band) {
        float sampled_value;
        if (outside_canvas) {
            sampled_value = nodata_value;
        } else {
            
            
            sampled_value = tex2D<float>(d_textures[band], src_x + 0.5f, src_y + 0.5f);
        }
        d_output[static_cast<size_t>(band) * max_chunk_pixels + dest_pixel_offset] = sampled_value;
    }
}



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
    cudaStream_t               stream) {

    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (dst_width  + 15) / 16,
        (dst_height + 15) / 16);

    
    const size_t shared_mem_bytes = 2 * WARP_GRID_WIDTH * WARP_GRID_HEIGHT * sizeof(float);

    kernel_warp<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        d_coarse_x, d_coarse_y, d_textures, d_output_flat,
        canvas_width, canvas_height,
        dst_width, dst_height,
        num_bands, nodata_value, max_chunk_pixels);
}
