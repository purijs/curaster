#include "../../include/raster_core.h"
#include "../../include/pipeline.h"
#include "../../include/cuda_atomic_float.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math_constants.h>




#define SORT_SWAP(a,b) { float _t = fminf(a,b); b = fmaxf(a,b); a = _t; }

__device__ __forceinline__ float median9(float v0, float v1, float v2,
                                          float v3, float v4, float v5,
                                          float v6, float v7, float v8)
{
    SORT_SWAP(v0,v1); SORT_SWAP(v3,v4); SORT_SWAP(v6,v7);
    SORT_SWAP(v1,v2); SORT_SWAP(v4,v5); SORT_SWAP(v7,v8);
    SORT_SWAP(v0,v1); SORT_SWAP(v3,v4); SORT_SWAP(v6,v7);
    SORT_SWAP(v0,v3); SORT_SWAP(v3,v6); SORT_SWAP(v0,v3);
    SORT_SWAP(v1,v4); SORT_SWAP(v4,v7); SORT_SWAP(v1,v4);
    SORT_SWAP(v2,v5); SORT_SWAP(v5,v8); SORT_SWAP(v2,v5);
    SORT_SWAP(v1,v3); SORT_SWAP(v5,v7);
    SORT_SWAP(v2,v6); SORT_SWAP(v4,v6); SORT_SWAP(v2,v4);
    SORT_SWAP(v3,v5); SORT_SWAP(v2,v3); SORT_SWAP(v5,v6);
    return v4;  
}
#undef SORT_SWAP





template<int RADIUS, int BX, int BY>
__global__ void kernel_focal(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int  src_width,
    int  halo_height,
    int  dst_width,
    int  dst_height,
    int  stat_id,
    int  shape_circle)   
{
    extern __shared__ float tile[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int out_x = blockIdx.x * BX + tx;
    int out_y = blockIdx.y * BY + ty;

    const int tile_w = BX + 2 * RADIUS;
    const int tile_h = BY + 2 * RADIUS;
    const int flat_id   = ty * BX + tx;
    const int tile_size = tile_w * tile_h;

    
    for (int i = flat_id; i < tile_size; i += BX * BY) {
        int tc = i % tile_w;
        int tr = i / tile_w;
        int sc = blockIdx.x * BX + tc - RADIUS;
        int sr = blockIdx.y * BY + tr - RADIUS;
        sc = max(0, min(src_width - 1,  sc));
        sr = max(0, min(halo_height - 1, sr));
        tile[i] = src[(size_t)sr * src_width + sc];
    }
    __syncthreads();

    if (out_x >= dst_width || out_y >= dst_height) return;

    const int LX = tx + RADIUS;
    const int LY = ty + RADIUS;
    const float R2 = (float)(RADIUS * RADIUS);

    
    float wf_mean = 0.f, wf_m2 = 0.f;
    float sum = 0.f, mn = FLT_MAX, mx = -FLT_MAX;
    int count = 0;

    
    if (RADIUS == 1 && stat_id == 6) {
        float v0 = tile[(LY-1)*tile_w + (LX-1)];
        float v1 = tile[(LY-1)*tile_w + (LX  )];
        float v2 = tile[(LY-1)*tile_w + (LX+1)];
        float v3 = tile[(LY  )*tile_w + (LX-1)];
        float v4 = tile[(LY  )*tile_w + (LX  )];
        float v5 = tile[(LY  )*tile_w + (LX+1)];
        float v6 = tile[(LY+1)*tile_w + (LX-1)];
        float v7 = tile[(LY+1)*tile_w + (LX  )];
        float v8 = tile[(LY+1)*tile_w + (LX+1)];
        dst[(size_t)out_y * dst_width + out_x] = median9(v0,v1,v2,v3,v4,v5,v6,v7,v8);
        return;
    }

    
    
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            if (shape_circle && ((float)(dx*dx + dy*dy) > R2)) continue;
            float v = tile[(LY + dy) * tile_w + (LX + dx)];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            ++count;
            float delta = v - wf_mean;
            wf_mean += delta / (float)count;
            wf_m2  += delta * (v - wf_mean);
        }
    }

    float n = (float)count;
    float result = 0.f;

    if (stat_id == 0)      result = (count > 0) ? wf_mean : 0.f;       
    else if (stat_id == 1) result = sum;                                  
    else if (stat_id == 2) result = mn;                                   
    else if (stat_id == 3) result = mx;                                   
    else if (stat_id == 4) result = (count > 1) ? sqrtf(wf_m2/(n-1.f)) : 0.f; 
    else if (stat_id == 5) result = (count > 1) ? wf_m2/(n-1.f) : 0.f;  
    else if (stat_id == 7) result = mx - mn;                              
    else if (stat_id == 6) {
        
        
        float range = mx - mn;
        if (range < 1e-10f) { result = mn; }
        else {
            const int L = 256;
            int hist[256] = {};
            for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
                for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                    if (shape_circle && ((float)(dx*dx+dy*dy) > R2)) continue;
                    float v = tile[(LY+dy)*tile_w + (LX+dx)];
                    int bin = (int)fminf((float)(L-1),
                                         fmaxf(0.f, (v - mn) / range * (float)L));
                    hist[bin]++;
                }
            }
            int target = (count + 1) / 2;
            int cum = 0;
            for (int b = 0; b < L; ++b) {
                cum += hist[b];
                if (cum >= target) {
                    result = mn + ((float)b + 0.5f) / (float)L * range;
                    break;
                }
            }
        }
    }

    dst[(size_t)out_y * dst_width + out_x] = result;
}


__global__ void kernel_focal_generic(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int  src_width,
    int  halo_height,
    int  dst_width,
    int  dst_height,
    int  radius,
    int  stat_id,
    int  shape_circle)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= dst_width || out_y >= dst_height) return;

    int halo_x = out_x + radius;
    int halo_y = out_y + radius;
    float R2 = (float)(radius * radius);

    float sum = 0.f, mn = FLT_MAX, mx = -FLT_MAX;
    float wf_mean = 0.f, wf_m2 = 0.f;
    int count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (shape_circle && (float)(dx*dx + dy*dy) > R2) continue;
            int sy = max(0, min(halo_height - 1, halo_y + dy));
            int sx = max(0, min(src_width - 1, halo_x + dx));
            float v = src[(size_t)sy * src_width + sx];
            sum += v; ++count;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            float delta = v - wf_mean;
            wf_mean += delta / (float)count;
            wf_m2  += delta * (v - wf_mean);
        }
    }

    float n = (float)count;
    float result = 0.f;
    if      (stat_id == 0) result = (count > 0) ? wf_mean : 0.f;
    else if (stat_id == 1) result = sum;
    else if (stat_id == 2) result = mn;
    else if (stat_id == 3) result = mx;
    else if (stat_id == 4) result = (count > 1) ? sqrtf(wf_m2/(n-1.f)) : 0.f;
    else if (stat_id == 5) result = (count > 1) ? wf_m2/(n-1.f) : 0.f;
    else if (stat_id == 7) result = mx - mn;

    dst[(size_t)out_y * dst_width + out_x] = result;
}



__global__ void kernel_terrain(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int     src_width,
    int     halo_height,
    int     dst_width,
    int     dst_height,
    uint32_t features_mask,
    float   cell_x,
    float   cell_y,
    float   sun_az_rad,
    float   sun_alt_rad,
    bool    use_zevenbergen,
    int     unit_mode,      
    size_t  max_chunk_pixels)
{
    const int BX = 32, BY = 16;
    __shared__ float tile[(BX+2)*(BY+2)];

    int tx = threadIdx.x, ty = threadIdx.y;
    int flat_id = ty * BX + tx;
    int tile_w  = BX + 2, tile_h = BY + 2;

    for (int i = flat_id; i < tile_w * tile_h; i += BX * BY) {
        int tc = i % tile_w;
        int tr = i / tile_w;
        int sc = max(0, min(src_width - 1,  (int)(blockIdx.x * BX + tc - 1)));
        int sr = max(0, min(halo_height - 1, (int)(blockIdx.y * BY + tr - 1)));
        tile[i] = src[(size_t)sr * src_width + sc];
    }
    __syncthreads();

    int out_x = blockIdx.x * BX + tx;
    int out_y = blockIdx.y * BY + ty;
    if (out_x >= dst_width || out_y >= dst_height) return;

    int lx = tx + 1, ly = ty + 1;
    float z1 = tile[(ly-1)*tile_w + (lx-1)];
    float z2 = tile[(ly-1)*tile_w + (lx  )];
    float z3 = tile[(ly-1)*tile_w + (lx+1)];
    float z4 = tile[(ly  )*tile_w + (lx-1)];
    float z5 = tile[(ly  )*tile_w + (lx  )];
    float z6 = tile[(ly  )*tile_w + (lx+1)];
    float z7 = tile[(ly+1)*tile_w + (lx-1)];
    float z8 = tile[(ly+1)*tile_w + (lx  )];
    float z9 = tile[(ly+1)*tile_w + (lx+1)];

    float p, q;
    if (use_zevenbergen) {
        p = (z6 - z4) / (2.f * cell_x);
        q = (z8 - z2) / (2.f * cell_y);
    } else {
        p = ((z3 + 2.f*z6 + z9) - (z1 + 2.f*z4 + z7)) / (8.f * cell_x);
        q = ((z7 + 2.f*z8 + z9) - (z1 + 2.f*z2 + z3)) / (8.f * cell_y);
    }

    float slope_rad  = atanf(sqrtf(p*p + q*q));
    float aspect_rad = atan2f(-q, p);
    if (aspect_rad < 0.f) aspect_rad += 2.f * CUDART_PI_F;

    
    float slope_out = slope_rad;
    if      (unit_mode == 0) slope_out = slope_rad * (180.f / CUDART_PI_F);
    else if (unit_mode == 2) slope_out = 100.f * tanf(slope_rad);

    size_t pixel    = (size_t)out_y * dst_width + out_x;
    int    band_idx = 0;

    if (features_mask & TERRAIN_SLOPE) {
        dst[band_idx * max_chunk_pixels + pixel] = slope_out;
        ++band_idx;
    }

    if (features_mask & TERRAIN_ASPECT) {
        float aspect_deg = 90.f - aspect_rad * (180.f / CUDART_PI_F);
        if (aspect_deg < 0.f)  aspect_deg += 360.f;
        if (slope_rad < 1e-6f) aspect_deg  = -9999.f;
        dst[band_idx * max_chunk_pixels + pixel] = aspect_deg;
        ++band_idx;
    }

    if (features_mask & TERRAIN_HILLSHADE) {
        float zenith = CUDART_PI_F / 2.f - sun_alt_rad;
        float hs = fmaxf(0.f, 255.f * (cosf(zenith) * cosf(slope_rad)
                    + sinf(zenith) * sinf(slope_rad) * cosf(sun_az_rad - aspect_rad)));
        dst[band_idx * max_chunk_pixels + pixel] = hs;
        ++band_idx;
    }

    if (features_mask & TERRAIN_TRI) {
        float tri = sqrtf((z1-z5)*(z1-z5) + (z2-z5)*(z2-z5) + (z3-z5)*(z3-z5)
                        + (z4-z5)*(z4-z5) + (z6-z5)*(z6-z5) + (z7-z5)*(z7-z5)
                        + (z8-z5)*(z8-z5) + (z9-z5)*(z9-z5));
        dst[band_idx * max_chunk_pixels + pixel] = tri;
        ++band_idx;
    }

    if (features_mask & TERRAIN_TPI) {
        dst[band_idx * max_chunk_pixels + pixel] = z5 - (z1+z2+z3+z4+z6+z7+z8+z9)/8.f;
        ++band_idx;
    }

    if (features_mask & TERRAIN_ROUGHNESS) {
        float mn9 = fminf(z1, fminf(z2, fminf(z3, fminf(z4, fminf(z5, fminf(z6, fminf(z7,fminf(z8,z9))))))));
        float mx9 = fmaxf(z1, fmaxf(z2, fmaxf(z3, fmaxf(z4, fmaxf(z5, fmaxf(z6, fmaxf(z7,fmaxf(z8,z9))))))));
        dst[band_idx * max_chunk_pixels + pixel] = mx9 - mn9;
        ++band_idx;
    }

    
    float cx2 = cell_x * cell_x, cy2 = cell_y * cell_y;
    float a_coef = ((z1+z3+z4+z6+z7+z9)/2.f - (z2+z5+z8)) / (3.f * cx2);
    float b_coef = (-z1+z3+z7-z9) / (4.f * cell_x * cell_y);
    float c_coef = ((z1+z2+z3+z7+z8+z9)/2.f - (z4+z5+z6)) / (3.f * cy2);
    float d_coef = (-z1-z4-z7+z3+z6+z9) / (6.f * cell_x);
    float e_coef = (z7+z8+z9 - z1-z2-z3) / (6.f * cell_y);
    float denom = d_coef*d_coef + e_coef*e_coef;

    if (features_mask & TERRAIN_PROF_CURV) {
        float prof = (denom > 1e-10f)
            ? -2.f*(a_coef*d_coef*d_coef + c_coef*e_coef*e_coef + b_coef*d_coef*e_coef)
              / (denom * denom)
            : 0.f;
        dst[band_idx * max_chunk_pixels + pixel] = prof;
        ++band_idx;
    }

    if (features_mask & TERRAIN_PLAN_CURV) {
        float plan = (denom > 1e-10f)
            ? -2.f*(a_coef*e_coef*e_coef - b_coef*d_coef*e_coef + c_coef*d_coef*d_coef) / denom
            : 0.f;
        dst[band_idx * max_chunk_pixels + pixel] = plan;
        ++band_idx;
    }

    if (features_mask & TERRAIN_TOTAL_CURV) {
        dst[band_idx * max_chunk_pixels + pixel] = 2.f * (a_coef + c_coef);
        ++band_idx;
    }
    (void)band_idx;
}



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
    cudaStream_t  stream)
{
    
    const int BX = 32, BY = 8;
    dim3 block(BX, BY);
    dim3 grid((dst_width  + BX - 1) / BX,
              (dst_height + BY - 1) / BY);

    auto smem = [&](int R) -> size_t {
        return (size_t)(BX + 2*R) * (BY + 2*R) * sizeof(float);
    };

#define DISPATCH_R(R) kernel_focal<R,BX,BY><<<grid,block,smem(R),stream>>>( \
    d_halo_src, d_output, src_width, halo_height, dst_width, dst_height, stat_id, shape_circle)

    if      (radius == 1)  { DISPATCH_R(1); }
    else if (radius == 2)  { DISPATCH_R(2); }
    else if (radius == 3)  { DISPATCH_R(3); }
    else if (radius == 5)  { DISPATCH_R(5); }
    else if (radius == 7)  { DISPATCH_R(7); }
    else if (radius == 10) { DISPATCH_R(10); }
    else if (radius == 15) { DISPATCH_R(15); }
    else {
        
        kernel_focal_generic<<<grid,block,0,stream>>>(
            d_halo_src, d_output, src_width, halo_height,
            dst_width, dst_height, radius, stat_id, shape_circle);
    }
#undef DISPATCH_R
}

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
    cudaStream_t  stream)
{
    dim3 block(32, 16);
    dim3 grid((dst_width  + 31) / 32,
              (dst_height + 15) / 16);
    size_t smem = (32+2) * (16+2) * sizeof(float);

    kernel_terrain<<<grid, block, smem, stream>>>(
        d_halo_src, d_output, src_width, halo_height,
        dst_width, dst_height, features_mask,
        cell_x, cell_y, sun_az_rad, sun_alt_rad,
        use_zevenbergen, unit_mode, max_chunk_pixels);
}
