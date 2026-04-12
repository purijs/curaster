/**
 * @file cuda_utils.h
 * @brief CUDA error-checking utility macro.
 */
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief Evaluate a CUDA API call and throw std::runtime_error on failure.
 *
 * Example:
 * @code
 *   CUDA_CHECK(cudaMalloc(&device_ptr, num_bytes));
 * @endcode
 */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t cuda_status = (call);                                 \
        if (cuda_status != cudaSuccess) {                                 \
            throw std::runtime_error(                                     \
                std::string("CUDA error: ") +                             \
                cudaGetErrorString(cuda_status));                         \
        }                                                                 \
    } while (0)
