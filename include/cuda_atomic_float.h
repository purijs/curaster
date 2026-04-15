#pragma once
#include <cuda_runtime.h>

__device__ inline void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(old);
        if (old_f <= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ inline void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(old);
        if (old_f >= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}
