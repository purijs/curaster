#include "../../include/raster_core.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math_constants.h>

#define GLCM_MAX_LEVELS 128






__global__ void kernel_glcm(
    const float* __restrict__ src,
    float*       __restrict__ dst,       
    int     src_width,
    int     halo_height,
    int     dst_width,
    int     dst_height,
    int     window,
    int     levels,
    float   val_min,
    float   val_range_inv,       
    int     dx,
    int     dy,
    int     feat_stride,         
    bool    log_scale)
{
    int warp_in_block = threadIdx.x / 32;
    int lane          = threadIdx.x % 32;
    int warps_per_blk = blockDim.x / 32;

    int out_x = blockIdx.x * warps_per_blk + warp_in_block;
    int out_y = blockIdx.y;
    if (out_x >= dst_width || out_y >= dst_height) return;

    int R  = window / 2;
    int hx = out_x + R;
    int hy = out_y + R;

    
    extern __shared__ char smem_raw[];
    int* glcm = (int*)smem_raw + warp_in_block * levels * levels;

    
    for (int i = lane; i < levels * levels; i += 32) {
        glcm[i] = 0;
    }
    __syncwarp();

    int W2 = window * window;

    
    for (int i = lane; i < W2; i += 32) {
        int wy = i / window, wx = i % window;
        int sy  = max(0, min(halo_height - 1, hy + (wy - R)));
        int sx  = max(0, min(src_width - 1,   hx + (wx - R)));
        int sy2 = max(0, min(halo_height - 1, sy + dy));
        int sx2 = max(0, min(src_width - 1,   sx + dx));

        float v1 = src[(size_t)sy  * src_width + sx ];
        float v2 = src[(size_t)sy2 * src_width + sx2];

        if (log_scale) {
            v1 = (v1 > 0.f) ? 10.f * log10f(v1) : -100.f;
            v2 = (v2 > 0.f) ? 10.f * log10f(v2) : -100.f;
        }

        int l1 = (int)fminf((float)(levels-1),
                             fmaxf(0.f, (v1 - val_min) * val_range_inv * (float)levels));
        int l2 = (int)fminf((float)(levels-1),
                             fmaxf(0.f, (v2 - val_min) * val_range_inv * (float)levels));

        atomicAdd(&glcm[l1 * levels + l2], 1);
        atomicAdd(&glcm[l2 * levels + l1], 1);  
    }
    __syncwarp();

    if (lane != 0) return;

    
    int total = 0;
    for (int i = 0; i < levels * levels; ++i) total += glcm[i];
    float inv_total = (total > 0) ? 1.f / (float)total : 1.f;

    
    float px[GLCM_MAX_LEVELS] = {}, py[GLCM_MAX_LEVELS] = {};
    for (int i = 0; i < levels; ++i) {
        for (int j = 0; j < levels; ++j) {
            float p = (float)glcm[i*levels+j] * inv_total;
            px[i] += p;
            py[j] += p;
        }
    }

    float mu_x = 0.f, mu_y = 0.f, sig_x = 0.f, sig_y = 0.f;
    for (int i = 0; i < levels; ++i) {
        mu_x += (float)i * px[i];
        mu_y += (float)i * py[i];
    }
    for (int i = 0; i < levels; ++i) {
        sig_x += (float)((i-mu_x)*(i-mu_x)) * px[i];
        sig_y += (float)((i-mu_y)*(i-mu_y)) * py[i];
    }
    sig_x = sqrtf(fmaxf(sig_x, 1e-10f));
    sig_y = sqrtf(fmaxf(sig_y, 1e-10f));

    
    float sum_av[2*GLCM_MAX_LEVELS] = {};
    float diff_av[GLCM_MAX_LEVELS]  = {};

    float asm_val = 0.f, contrast = 0.f, corr_num = 0.f;
    float homogeneity = 0.f, entropy = 0.f, dissimilarity = 0.f;
    float autocorr = 0.f, max_prob = 0.f, variance = 0.f;
    float cluster_shade = 0.f, cluster_prom = 0.f;

    for (int i = 0; i < levels; ++i) {
        for (int j = 0; j < levels; ++j) {
            float p = (float)glcm[i*levels+j] * inv_total;
            int diff = abs(i - j);
            float s  = (float)(i + j) - mu_x - mu_y;

            asm_val      += p * p;
            contrast     += (float)(diff*diff) * p;
            corr_num     += (float)(i*j) * p;
            homogeneity  += p / (1.f + (float)(diff*diff));
            dissimilarity += (float)diff * p;
            autocorr     += (float)(i*j) * p;
            variance += ((float)i - mu_x)*((float)i - mu_x) * p;
            entropy  -= (p > 1e-10f) ? p * log2f(p) : 0.f;
            max_prob  = fmaxf(max_prob, p);
            cluster_shade += s*s*s * p;
            cluster_prom  += s*s*s*s * p;

            int si = i + j;
            if (si < 2*levels) sum_av[si]  += p;
            if (diff < levels) diff_av[diff] += p;
        }
    }

    float correlation = (sig_x * sig_y > 1e-10f)
        ? (corr_num - mu_x * mu_y) / (sig_x * sig_y)
        : 0.f;

    float sum_average = 0.f, sum_variance = 0.f, sum_entropy = 0.f;
    float diff_variance = 0.f, diff_entropy = 0.f;
    for (int k = 0; k < 2*levels; ++k) {
        sum_average += (float)k * sum_av[k];
    }
    for (int k = 0; k < 2*levels; ++k) {
        float v = sum_av[k];
        sum_variance += ((float)k - sum_average)*((float)k - sum_average) * v;
        if (v > 1e-10f) sum_entropy -= v * log2f(v);
    }
    for (int k = 0; k < levels; ++k) {
        diff_variance += (float)(k*k) * diff_av[k];
        float v = diff_av[k];
        if (v > 1e-10f) diff_entropy -= v * log2f(v);
    }

    
    float HX = 0.f, HY = 0.f, HXY1 = 0.f, HXY2 = 0.f;
    for (int i = 0; i < levels; ++i) {
        if (px[i] > 1e-10f) HX -= px[i] * log2f(px[i]);
        if (py[i] > 1e-10f) HY -= py[i] * log2f(py[i]);
    }
    for (int i = 0; i < levels; ++i) {
        for (int j = 0; j < levels; ++j) {
            float p = (float)glcm[i*levels+j] * inv_total;
            float pij = px[i] * py[j] + 1e-10f;
            HXY1 -= p * log2f(pij);
            HXY2 -= pij * log2f(pij);
        }
    }
    float maxH = fmaxf(HX, HY);
    float imc1 = (maxH > 1e-10f) ? (entropy - HXY1) / maxH : 0.f;
    float imc2_inner = 1.f - expf(-2.f * (HXY2 - entropy));
    float imc2 = (imc2_inner > 0.f) ? sqrtf(imc2_inner) : 0.f;

    
    size_t pixel = (size_t)out_y * dst_width + out_x;
    const float features[18] = {
        asm_val, contrast, correlation, variance, homogeneity,
        sum_average, sum_variance, sum_entropy, entropy,
        diff_variance, diff_entropy, dissimilarity, autocorr,
        max_prob, cluster_shade, cluster_prom, imc1, imc2
    };
    for (int f = 0; f < 18; ++f) {
        dst[(size_t)f * feat_stride + pixel] = features[f];
    }
}




__global__ void kernel_glcm_avg_divide(
    float* __restrict__ dst,
    int    num_pixels,
    int    num_features,
    float  inv_num_dirs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;
    for (int f = 0; f < num_features; ++f) {
        dst[(size_t)f * num_pixels + idx] *= inv_num_dirs;
    }
}



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
    cudaStream_t  stream)
{
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    dim3 block(threads, 1);
    dim3 grid((dst_width + warps_per_block - 1) / warps_per_block, dst_height);

    float range = (val_max > val_min) ? (val_max - val_min) : 1.f;
    float range_inv = 1.f / range;

    size_t smem = (size_t)warps_per_block * levels * levels * sizeof(int);

    kernel_glcm<<<grid, block, smem, stream>>>(
        d_halo_src, d_output, src_width, halo_height,
        dst_width, dst_height, window, levels,
        val_min, range_inv, dx, dy,
        (int)max_chunk_pixels, log_scale);
    (void)num_output_features;
}

void launch_glcm_avg_divide(
    float*        d_output,
    int           num_pixels,
    int           num_features,
    int           num_dirs,
    cudaStream_t  stream)
{
    const int kThreads = 256;
    int nblocks = (num_pixels + kThreads - 1) / kThreads;
    float inv = 1.f / (float)num_dirs;
    kernel_glcm_avg_divide<<<nblocks, kThreads, 0, stream>>>(
        d_output, num_pixels, num_features, inv);
}
