/**
 * @file chunk_queue.h
 * @brief Thread-safe bounded queue for streaming raster chunks to callers.
 *
 * ChunkQueue lets the processing engine push completed chunks from a background
 * thread while a Python caller pops them one at a time via Chain::iter_begin().
 * A capacity cap prevents unbounded memory growth if the consumer is slow.
 *

 * The blocking push() is preserved for callers that want the old behaviour.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>


/**
 * @brief One completed horizontal slice of output pixels.
 */
struct ChunkResult {
    std::vector<float> data;
    int                width;
    int                height;
    int                y_offset;
};


/**
 * @brief A bounded, thread-safe FIFO queue of ChunkResult objects.
 *
 * The producer (engine thread) calls push() or push_with_timeout(); the
 * consumer (Python thread) calls pop(). When the producer finishes it calls
 * finish() so pop() can signal EOF to the caller by returning false.
 */
class ChunkQueue {
public:
    /// Construct with a maximum queue depth of @p capacity chunks.
    explicit ChunkQueue(int capacity) : capacity_(capacity) {}

    
    /**
     * @brief Push a chunk onto the queue.
     *
     * Blocks if the queue is already at capacity, preventing unbounded RAM use.
     * Use push_with_timeout() if the caller needs to remain responsive.
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
     * @brief Try to push a chunk, waiting at most @p timeout_ms milliseconds.
     *
     * Returns true if the chunk was enqueued, false if the queue was still full
     * after the timeout.  The engine's queue_callback should call this and, on
     * false, synchronise its current CUDA stream and then retry \u2014 giving the
     * GPU a chance to hand off work rather than blocking indefinitely on pinned
     * memory that cannot be released until the consumer pops.
     *
     * @param chunk       Chunk to enqueue (moved into queue on success).
     * @param timeout_ms  Maximum wait in milliseconds (default 100 ms).
     * @return true if enqueued; false if timed out.
     */
    bool push_with_timeout(ChunkResult chunk,
                           int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool accepted = condition_.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [this] { return static_cast<int>(queue_.size()) < capacity_ || done_; });
        if (!accepted) {
            return false;  
        }
        queue_.push(std::move(chunk));
        condition_.notify_all();
        return true;
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

    

    /// Return the number of chunks currently in the queue.
    int size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return static_cast<int>(queue_.size());
    }

    /// Return the maximum queue depth.
    int capacity() const {
        return capacity_;
    }

    /// Return a value in [0.0, 1.0] representing how full the queue is.
    /// The engine can use this to throttle prefetch rate before the queue fills.
    double utilization() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return capacity_ > 0
                   ? static_cast<double>(queue_.size()) / capacity_
                   : 0.0;
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable condition_;
    std::queue<ChunkResult> queue_;
    bool                    done_     = false;
    int                     capacity_;
};
