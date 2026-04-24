#include "../../include/raster_core.h"
#include "../../include/cuda_atomic_float.h"
#include <cuda_runtime.h>
#include <float.h>




__global__ void kernel_zonal_reduction(
    const float*    __restrict__ d_values,
    const uint16_t* __restrict__ d_zone_labels,
    size_t          num_pixels,
    int*            d_count,
    float*          d_sum,
    float*          d_sum_sq,
    float*          d_min,
    float*          d_max,
    int             num_zones)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    uint16_t zone_id = 0;
    float val = 0.f;
    bool valid = false;

    if (idx < num_pixels) {
        zone_id = d_zone_labels[idx];
        if (zone_id > 0 && zone_id <= (uint16_t)num_zones) {
            val   = d_values[idx];
            valid = true;
        }
    }

    
    
    
    
    
    
    
    

    if (valid) {
        
        unsigned mask = __activemask();

        
        
        
        
        
        for (int step = 0; step < 32; ++step) {
            uint16_t bcast_id = __shfl_sync(mask, zone_id, step);
            if (!__shfl_sync(mask, (int)valid, step)) continue; 

            if ((int)(threadIdx.x & 31) == step) {
                
                
                float local_sum = val, local_sum_sq = val*val;
                float local_min = val, local_max = val;
                int   local_cnt = 1;

                for (int l = step+1; l < 32; ++l) {
                    uint16_t peer_id = __shfl_sync(mask, zone_id, l);
                    bool peer_valid  = (bool)__shfl_sync(mask, (int)valid, l);
                    if (peer_valid && peer_id == bcast_id) {
                        float pval = __shfl_sync(mask, val, l);
                        local_sum    += pval;
                        local_sum_sq += pval * pval;
                        local_min     = fminf(local_min, pval);
                        local_max     = fmaxf(local_max, pval);
                        ++local_cnt;
                    }
                }
                atomicAdd(&d_count [bcast_id], local_cnt);
                atomicAdd(&d_sum   [bcast_id], local_sum);
                atomicAdd(&d_sum_sq[bcast_id], local_sum_sq);
                atomicMinFloat(&d_min[bcast_id], local_min);
                atomicMaxFloat(&d_max[bcast_id], local_max);
            }
            
            
            if ((int)(threadIdx.x & 31) > step && zone_id == bcast_id && valid) {
                valid = false;  
            }
        }
    }
}






__global__ void kernel_temporal(
    const float** __restrict__ d_scene_ptrs,
    float*        __restrict__ d_output,
    size_t        num_pixels,
    int           num_scenes,
    int           op_id,
    int           t0_idx,
    int           t1_idx,
    const float*  __restrict__ d_time_values,
    float         denominator)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    if (op_id == 0) { 
        d_output[idx] = d_scene_ptrs[t1_idx][idx] - d_scene_ptrs[t0_idx][idx];

    } else if (op_id == 1) { 
        d_output[idx] = d_scene_ptrs[t1_idx][idx] / (d_scene_ptrs[t0_idx][idx] + 1e-6f);

    } else if (op_id == 2) { 
        float mean = 0.f;
        for (int s = 0; s < num_scenes; ++s) mean += d_scene_ptrs[s][idx];
        mean /= (float)num_scenes;
        
        d_output[idx] = d_scene_ptrs[t0_idx][idx] - mean;

    } else if (op_id == 3) { 
        d_output[idx] = d_scene_ptrs[t0_idx][idx] - d_scene_ptrs[t1_idx][idx];

    } else if (op_id == 4) { 
        float sum_t = 0.f, sum_v = 0.f, sum_tv = 0.f;
        for (int s = 0; s < num_scenes; ++s) {
            float t = d_time_values[s];
            float v = d_scene_ptrs[s][idx];
            sum_t  += t; sum_v += v; sum_tv += t * v;
        }
        float n = (float)num_scenes;
        float numerator = n * sum_tv - sum_t * sum_v;
        d_output[idx] = (fabsf(denominator) > 1e-10f) ? numerator / denominator : 0.f;

    } else if (op_id == 5) { 
        float sum = 0.f;
        for (int s = 0; s < num_scenes; ++s) sum += d_scene_ptrs[s][idx];
        d_output[idx] = sum / (float)num_scenes;

    } else if (op_id == 6) { 
        float sum = 0.f, sum_sq = 0.f;
        for (int s = 0; s < num_scenes; ++s) {
            float v = d_scene_ptrs[s][idx];
            sum += v; sum_sq += v * v;
        }
        float n = (float)num_scenes;
        d_output[idx] = sqrtf(fmaxf(0.f, sum_sq/n - (sum/n)*(sum/n)));

    } else if (op_id == 7) { 
        float mn = FLT_MAX;
        for (int s = 0; s < num_scenes; ++s) mn = fminf(mn, d_scene_ptrs[s][idx]);
        d_output[idx] = mn;

    } else if (op_id == 8) { 
        float mx = -FLT_MAX;
        for (int s = 0; s < num_scenes; ++s) mx = fmaxf(mx, d_scene_ptrs[s][idx]);
        d_output[idx] = mx;
    }
}



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
    cudaStream_t    stream)
{
    const int kThreads = 256;
    int nblocks = static_cast<int>((num_pixels + kThreads - 1) / kThreads);
    kernel_zonal_reduction<<<nblocks, kThreads, 0, stream>>>(
        d_values, d_zone_labels, num_pixels,
        d_count, d_sum, d_sum_sq, d_min, d_max, num_zones);
}

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
    cudaStream_t  stream)
{
    const int kThreads = 256;
    int nblocks = static_cast<int>((num_pixels + kThreads - 1) / kThreads);
    kernel_temporal<<<nblocks, kThreads, 0, stream>>>(
        d_scene_ptrs, d_output, num_pixels, num_scenes,
        op_id, t0_idx, t1_idx, d_time_values, denominator);
}
