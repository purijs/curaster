/**
 * @file gpu.cu
 * @brief GPU kernels for raster algebra and polygon mask application.
 *
 * Two kernels are provided:
 *
 *  - kernel_raster_algebra: Executes the compiled algebra VM instruction
 *    sequence on every output pixel using a float4 vectorised main loop
 *    (processes 4 pixels per thread per iteration) plus a scalar tail loop
 *    for pixels that do not fill a complete float4.
 *
 *  - kernel_apply_mask: Zeros out any pixel that falls outside all polygon
 *    spans for its row (post-process the algebra output in-place).
 */
#include "../../include/raster_core.h"
#include <cuda_runtime.h>

// ─── float4 component-wise helpers ───────────────────────────────────────────
// These inline device functions mirror the scalar VM operations but operate
// on four pixels simultaneously, enabling 4× throughput per thread.

__device__ inline float4 f4_add(float4 a, float4 b) {
    return make_float4(a.x + b.x,
                       a.y + b.y,
                       a.z + b.z,
                       a.w + b.w);
}

__device__ inline float4 f4_sub(float4 a, float4 b) {
    return make_float4(a.x - b.x,
                       a.y - b.y,
                       a.z - b.z,
                       a.w - b.w);
}

__device__ inline float4 f4_mul(float4 a, float4 b) {
    return make_float4(a.x * b.x,
                       a.y * b.y,
                       a.z * b.z,
                       a.w * b.w);
}

/// Safe division: adds a small epsilon to the denominator to prevent NaN/Inf.
__device__ inline float4 f4_div(float4 a, float4 b) {
    return make_float4(__fdividef(a.x, b.x + 1e-6f),
                       __fdividef(a.y, b.y + 1e-6f),
                       __fdividef(a.z, b.z + 1e-6f),
                       __fdividef(a.w, b.w + 1e-6f));
}

// ── Comparison operators — return 1.0 for true, 0.0 for false ────────────────
__device__ inline float4 f4_gt(float4 a, float4 b) {
    return make_float4(a.x >  b.x ? 1.f : 0.f,
                       a.y >  b.y ? 1.f : 0.f,
                       a.z >  b.z ? 1.f : 0.f,
                       a.w >  b.w ? 1.f : 0.f);
}

__device__ inline float4 f4_lt(float4 a, float4 b) {
    return make_float4(a.x <  b.x ? 1.f : 0.f,
                       a.y <  b.y ? 1.f : 0.f,
                       a.z <  b.z ? 1.f : 0.f,
                       a.w <  b.w ? 1.f : 0.f);
}

__device__ inline float4 f4_gte(float4 a, float4 b) {
    return make_float4(a.x >= b.x ? 1.f : 0.f,
                       a.y >= b.y ? 1.f : 0.f,
                       a.z >= b.z ? 1.f : 0.f,
                       a.w >= b.w ? 1.f : 0.f);
}

__device__ inline float4 f4_lte(float4 a, float4 b) {
    return make_float4(a.x <= b.x ? 1.f : 0.f,
                       a.y <= b.y ? 1.f : 0.f,
                       a.z <= b.z ? 1.f : 0.f,
                       a.w <= b.w ? 1.f : 0.f);
}

__device__ inline float4 f4_eq(float4 a, float4 b) {
    return make_float4(a.x == b.x ? 1.f : 0.f,
                       a.y == b.y ? 1.f : 0.f,
                       a.z == b.z ? 1.f : 0.f,
                       a.w == b.w ? 1.f : 0.f);
}

__device__ inline float4 f4_neq(float4 a, float4 b) {
    return make_float4(a.x != b.x ? 1.f : 0.f,
                       a.y != b.y ? 1.f : 0.f,
                       a.z != b.z ? 1.f : 0.f,
                       a.w != b.w ? 1.f : 0.f);
}

// ── Logical operators (fuzzy: AND = multiply, OR = max) ───────────────────────
__device__ inline float4 f4_and(float4 a, float4 b) {
    return f4_mul(a, b);  // Fuzzy AND: truth degree = product
}

__device__ inline float4 f4_or(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x),
                       fmaxf(a.y, b.y),
                       fmaxf(a.z, b.z),
                       fmaxf(a.w, b.w));
}

__device__ inline float4 f4_not(float4 a) {
    return f4_sub(make_float4(1.f, 1.f, 1.f, 1.f), a);
}

/// Conditional blend: result = cond * true_val + (1 - cond) * false_val
__device__ inline float4 f4_if(float4 condition, float4 true_val, float4 false_val) {
    return f4_add(f4_mul(condition, true_val),
                  f4_mul(f4_not(condition), false_val));
}

// ── min / max ─────────────────────────────────────────────────────────────────
__device__ inline float4 f4_min(float4 a, float4 b) {
    return make_float4(fminf(a.x, b.x),
                       fminf(a.y, b.y),
                       fminf(a.z, b.z),
                       fminf(a.w, b.w));
}

__device__ inline float4 f4_max(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x),
                       fmaxf(a.y, b.y),
                       fmaxf(a.z, b.z),
                       fmaxf(a.w, b.w));
}

// ─── kernel_raster_algebra ────────────────────────────────────────────────────
/**
 * @brief Evaluate the algebra VM over all pixels.
 *
 * Main loop: processes 4 pixels per iteration using float4 loads.
 * Tail loop: handles any pixels at the end that do not fill a float4.
 *
 * Stack depth is fixed at 24 — sufficient for any practical expression.
 */
__global__ void kernel_raster_algebra(
    const Instruction* __restrict__ instructions,
    int                             num_instructions,
    const float* const* __restrict__ bands,
    float* __restrict__             output,
    size_t                          num_pixels) {

    // ── Vectorised (float4) main loop ─────────────────────────────────────
    size_t base_pixel  = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x * 4;

    while (base_pixel + 3 < num_pixels) {
        float4 stack[24];
        int    stack_top = 0;

        for (int inst_idx = 0; inst_idx < num_instructions; ++inst_idx) {
            const Instruction& inst = instructions[inst_idx];

            if (inst.op == OP_LOAD_CONST) {
                float val = inst.constant;
                stack[stack_top++] = make_float4(val, val, val, val);

            } else if (inst.op == OP_LOAD_BAND) {
                stack[stack_top++] = __ldg(
                    reinterpret_cast<const float4*>(bands[inst.band_index] + base_pixel));

            } else if (inst.op == OP_ADD) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_add(lhs, rhs);

            } else if (inst.op == OP_SUB) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_sub(lhs, rhs);

            } else if (inst.op == OP_MUL) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_mul(lhs, rhs);

            } else if (inst.op == OP_DIV) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_div(lhs, rhs);

            } else if (inst.op == OP_GT) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_gt(lhs, rhs);

            } else if (inst.op == OP_LT) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_lt(lhs, rhs);

            } else if (inst.op == OP_GTE) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_gte(lhs, rhs);

            } else if (inst.op == OP_LTE) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_lte(lhs, rhs);

            } else if (inst.op == OP_EQ) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_eq(lhs, rhs);

            } else if (inst.op == OP_NEQ) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_neq(lhs, rhs);

            } else if (inst.op == OP_AND) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_and(lhs, rhs);

            } else if (inst.op == OP_OR) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_or(lhs, rhs);

            } else if (inst.op == OP_NOT) {
                stack[stack_top - 1] = f4_not(stack[stack_top - 1]);

            } else if (inst.op == OP_IF) {
                float4 false_val  = stack[--stack_top];
                float4 true_val   = stack[--stack_top];
                float4 condition  = stack[--stack_top];
                stack[stack_top++] = f4_if(condition, true_val, false_val);

            } else if (inst.op == OP_BETWEEN) {
                float4 hi = stack[--stack_top];
                float4 lo = stack[--stack_top];
                float4 x  = stack[--stack_top];
                stack[stack_top++] = f4_and(f4_gte(x, lo), f4_lte(x, hi));

            } else if (inst.op == OP_CLAMP) {
                float4 hi = stack[--stack_top];
                float4 lo = stack[--stack_top];
                float4 x  = stack[--stack_top];
                stack[stack_top++] = f4_max(lo, f4_min(x, hi));

            } else if (inst.op == OP_MIN2) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_min(lhs, rhs);

            } else if (inst.op == OP_MAX2) {
                float4 rhs = stack[--stack_top];
                float4 lhs = stack[--stack_top];
                stack[stack_top++] = f4_max(lhs, rhs);
            }
        }

        // Write 4 results at once.
        *reinterpret_cast<float4*>(output + base_pixel) =
            (stack_top > 0) ? stack[0] : make_float4(0.f, 0.f, 0.f, 0.f);

        base_pixel += grid_stride;
    }

    // ── Scalar tail loop ──────────────────────────────────────────────────
    // Handles remaining pixels (num_pixels % 4 != 0 case).
    size_t tail_pixel = (num_pixels / 4) * 4
                      + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tail_pixel < num_pixels) {
        float stack[24];
        int   stack_top = 0;

        for (int inst_idx = 0; inst_idx < num_instructions; ++inst_idx) {
            const Instruction& inst = instructions[inst_idx];

            if (inst.op == OP_LOAD_CONST) {
                stack[stack_top++] = inst.constant;

            } else if (inst.op == OP_LOAD_BAND) {
                stack[stack_top++] = __ldg(&bands[inst.band_index][tail_pixel]);

            } else if (inst.op == OP_ADD) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = lhs + rhs;

            } else if (inst.op == OP_SUB) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = lhs - rhs;

            } else if (inst.op == OP_MUL) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = lhs * rhs;

            } else if (inst.op == OP_DIV) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = __fdividef(lhs, rhs + 1e-6f);

            } else if (inst.op == OP_GT) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = (lhs >  rhs) ? 1.f : 0.f;

            } else if (inst.op == OP_LT) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = (lhs <  rhs) ? 1.f : 0.f;

            } else if (inst.op == OP_GTE) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = (lhs >= rhs) ? 1.f : 0.f;

            } else if (inst.op == OP_LTE) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = (lhs <= rhs) ? 1.f : 0.f;

            } else if (inst.op == OP_EQ) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = (lhs == rhs) ? 1.f : 0.f;

            } else if (inst.op == OP_NEQ) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = (lhs != rhs) ? 1.f : 0.f;

            } else if (inst.op == OP_AND) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = lhs * rhs;

            } else if (inst.op == OP_OR) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = fmaxf(lhs, rhs);

            } else if (inst.op == OP_NOT) {
                stack[stack_top - 1] = 1.f - stack[stack_top - 1];

            } else if (inst.op == OP_IF) {
                float false_val = stack[--stack_top];
                float true_val  = stack[--stack_top];
                float condition = stack[--stack_top];
                stack[stack_top++] = condition * true_val + (1.f - condition) * false_val;

            } else if (inst.op == OP_BETWEEN) {
                float hi = stack[--stack_top];
                float lo = stack[--stack_top];
                float x  = stack[--stack_top];
                stack[stack_top++] = (x >= lo && x <= hi) ? 1.f : 0.f;

            } else if (inst.op == OP_CLAMP) {
                float hi = stack[--stack_top];
                float lo = stack[--stack_top];
                float x  = stack[--stack_top];
                stack[stack_top++] = fmaxf(lo, fminf(x, hi));

            } else if (inst.op == OP_MIN2) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = fminf(lhs, rhs);

            } else if (inst.op == OP_MAX2) {
                float rhs = stack[--stack_top];
                float lhs = stack[--stack_top];
                stack[stack_top++] = fmaxf(lhs, rhs);
            }
        }

        output[tail_pixel] = (stack_top > 0) ? stack[0] : 0.f;
    }
}

// ─── kernel_apply_mask ────────────────────────────────────────────────────────
/**
 * @brief Zero out pixels that fall outside all polygon spans for their row.
 *
 * Each thread handles one pixel. The thread maps its flat pixel index to
 * (row, col), then checks all spans for that row.
 */
__global__ void kernel_apply_mask(
    float* __restrict__            output,
    size_t                         num_pixels,
    int                            image_width,
    const GpuSpanRow* __restrict__ spans) {

    size_t pixel_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) { return; }

    int image_row = static_cast<int>(pixel_idx / static_cast<size_t>(image_width));
    int image_col = static_cast<int>(pixel_idx % static_cast<size_t>(image_width));

    const GpuSpanRow& span_row = spans[image_row];
    int inside = 0;

    for (int span_idx = 0; span_idx < span_row.num_spans; ++span_idx) {
        if (image_col >= span_row.spans[span_idx].start
         && image_col <= span_row.spans[span_idx].end) {
            inside = 1;
            break;
        }
    }

    output[pixel_idx] *= static_cast<float>(inside);
}

// ─── Kernel launch wrappers ───────────────────────────────────────────────────

void launch_raster_algebra(
    const Instruction*  d_instructions,
    int                 num_instructions,
    const float* const* d_bands,
    float*              d_output,
    size_t              num_pixels,
    cudaStream_t        stream) {

    const int kThreadsPerBlock = 256;

    // Each thread processes 4 pixels, so we need ceil(num_pixels/4) thread groups.
    size_t pixel_groups = (num_pixels + 3) / 4;
    int    num_blocks   = static_cast<int>((pixel_groups + kThreadsPerBlock - 1) / kThreadsPerBlock);

    kernel_raster_algebra<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        d_instructions, num_instructions, d_bands, d_output, num_pixels);
}

void launch_apply_mask(
    float*            d_output,
    size_t            num_pixels,
    int               image_width,
    const GpuSpanRow* d_spans,
    cudaStream_t      stream) {

    if (num_pixels == 0 || d_spans == nullptr) { return; }

    const int kThreadsPerBlock = 256;
    int num_blocks = static_cast<int>((num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock);

    kernel_apply_mask<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        d_output, num_pixels, image_width, d_spans);
}
