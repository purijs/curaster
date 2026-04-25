#pragma once

#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cuda_utils.h"

class PinnedArena {
public:
    void* alloc(size_t bytes, size_t align = 256) {
        if (bytes == 0) return nullptr;
        size_t aligned_used = (used_ + align - 1) & ~(align - 1);
        if (aligned_used + bytes > capacity_) {
            throw std::runtime_error(
                "PinnedArena: out of pinned memory (" +
                std::to_string((aligned_used + bytes) / 1048576) + " MB requested, " +
                std::to_string(capacity_ / 1048576) + " MB capacity)");
        }
        void* ptr = static_cast<char*>(base_) + aligned_used;
        used_ = aligned_used + bytes;
        return ptr;
    }

    template<typename T>
    T* alloc_typed(size_t count, size_t align = 256) {
        return static_cast<T*>(alloc(count * sizeof(T), align));
    }

    void reset() {
        used_ = 0;
    }

    
    
    void release() {
        if (base_) { cudaFreeHost(base_); base_ = nullptr; }
        capacity_ = 0;
        used_     = 0;
    }

    void ensure(size_t min_bytes) {
        if (base_ && capacity_ >= min_bytes) return;
        if (base_) {
            cudaFreeHost(base_);
            base_     = nullptr;
            capacity_ = 0;
            used_     = 0;
        }
        CUDA_CHECK(cudaHostAlloc(&base_, min_bytes,
                                 cudaHostAllocMapped | cudaHostAllocWriteCombined));
        capacity_ = min_bytes;
        used_     = 0;
    }

    void* base() const { return base_; }
    size_t capacity() const { return capacity_; }
    size_t used() const { return used_; }

    ~PinnedArena() {
        if (base_) cudaFreeHost(base_);
    }

private:
    void*  base_     = nullptr;
    size_t capacity_ = 0;
    size_t used_     = 0;
};

extern PinnedArena g_pinned_arena;
