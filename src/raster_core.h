#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>

enum Opcode {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_LOAD_BAND, OP_LOAD_CONST,
    OP_GT, OP_LT, OP_GTE, OP_LTE, OP_EQ, OP_NEQ,
    OP_AND, OP_OR, OP_NOT,
    OP_IF, OP_BETWEEN, OP_CLAMP,
    OP_MIN2, OP_MAX2
};

struct Instruction { 
    Opcode op; 
    float constant; 
    int band_index; 
};

void launch_raster_algebra(
    const Instruction* d_prog, 
    int np,
    const float* const* d_bands,
    float* d_out, 
    size_t sz,
    cudaStream_t stream
);