/**
 * @file chunk_queue.h
 * @brief Thread-safe bounded queue for streaming raster chunks to callers.
 *
 * ChunkQueue lets the processing engine push completed chunks from a background
 * thread while a Python caller pops them one at a time via Chain::iter_begin().
 * A capacity cap prevents unbounded memory growth if the consumer is slow.
 */
#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

// ─── Single processed chunk ───────────────────────────────────────────────────
/**
 * @brief One completed horizontal slice of output pixels.
 */
struct ChunkResult {
    std::vector<float> data;   ///< Pixel values in row-major order
    int                width;  ///< Chunk width  in pixels (equals image width)
    int                height; ///< Chunk height in pixels
    int                y_offset; ///< Top row of this chunk in the full output raster
};

// ─── Bounded thread-safe chunk queue ─────────────────────────────────────────
/**
 * @brief A bounded, thread-safe FIFO queue of ChunkResult objects.
 *
 * The producer (engine thread) calls push(); the consumer (Python thread) calls
 * pop(). When the producer finishes it calls finish() so pop() can signal EOF
 * to the caller by returning false.
 */
class ChunkQueue {
public:
    /// Construct with a maximum queue depth of @p capacity chunks.
    explicit ChunkQueue(int capacity) : capacity_(capacity) {}

    /**
     * @brief Push a chunk onto the queue.
     *
     * Blocks if the queue is already at capacity, preventing unbounded RAM use.
     */
    void push(ChunkResult chunk) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
            return static_cast<int>(queue_.size()) < capacity_ || done_;
        });
        queue_.push(std::move(chunk));
        condition_.notify_all();
    }

    /**
     * @brief Pop the next chunk from the queue.
     *
     * Blocks until a chunk is available or finish() has been called.
     * @return true if @p out was filled, false if the queue is exhausted.
     */
    bool pop(ChunkResult& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop();
        condition_.notify_all();
        return true;
    }

    /// Signal that no more chunks will be pushed; unblocks any waiting pop().
    void finish() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_ = true;
        condition_.notify_all();
    }

private:
    std::queue<ChunkResult>  queue_;
    std::mutex               mutex_;
    std::condition_variable  condition_;
    bool                     done_     = false;
    int                      capacity_;
};
