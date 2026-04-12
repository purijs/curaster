/**
 * @file s3_fetch.cpp
 * @brief Parallel S3 tile fetcher with request merging via libcurl multi.
 */
#include "s3_fetch.h"
#include "s3_auth.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <curl/curl.h>

// ─── libcurl write callback context ──────────────────────────────────────────

/// Context passed to the libcurl write callback — holds a pointer to the
/// destination byte buffer for the current request.
struct CurlWriteContext {
    std::vector<uint8_t>* buffer; ///< Destination for received bytes
};

/// libcurl CURLOPT_WRITEFUNCTION implementation.
/// Named write_response_bytes to avoid clashing with libcurl's own
/// curl_write_callback typedef defined in <curl/curl.h>.
static size_t write_response_bytes(void* received_data,
                                   size_t element_size,
                                   size_t num_elements,
                                   void*  user_context) {
    size_t total_bytes  = element_size * num_elements;
    auto*  write_ctx    = static_cast<CurlWriteContext*>(user_context);
    auto*  dest_buffer  = write_ctx->buffer;
    auto*  byte_ptr     = static_cast<uint8_t*>(received_data);

    dest_buffer->insert(dest_buffer->end(), byte_ptr, byte_ptr + total_bytes);
    return total_bytes;
}

// ─── Merged byte-range request ───────────────────────────────────────────────

/// One HTTP Range request that may cover multiple individual tile fetch jobs.
struct MergedRangeRequest {
    size_t           first_byte;  ///< Start of the merged byte range
    size_t           last_byte;   ///< End   of the merged byte range (exclusive)
    std::vector<int> job_indices; ///< Indices into the jobs[] array covered by this request
};

// ─── s3_fetch_tiles ──────────────────────────────────────────────────────────

void s3_fetch_tiles(const S3Loc&               location,
                    const std::vector<size_t>&  all_offsets,
                    const std::vector<size_t>&  all_lengths,
                    std::vector<TileFetch>&      jobs) {

    // Sort jobs by byte offset so adjacent ones can be merged.
    std::vector<int> sorted_job_indices(jobs.size());
    std::iota(sorted_job_indices.begin(), sorted_job_indices.end(), 0);
    std::sort(sorted_job_indices.begin(), sorted_job_indices.end(),
              [&](int a, int b) {
                  return jobs[a].byte_offset < jobs[b].byte_offset;
              });

    // Merge jobs whose byte ranges are within 128 KB of each other.
    const size_t merge_gap_bytes = 128 * 1024;
    std::vector<MergedRangeRequest> merged_requests;

    for (int job_index : sorted_job_indices) {
        TileFetch& job = jobs[job_index];
        size_t     job_end = job.byte_offset + job.byte_length;

        bool merge_with_last = !merged_requests.empty()
                            && job.byte_offset <= merged_requests.back().last_byte + merge_gap_bytes;

        if (merge_with_last) {
            merged_requests.back().last_byte = (std::max)(merged_requests.back().last_byte, job_end);
            merged_requests.back().job_indices.push_back(job_index);
        } else {
            MergedRangeRequest new_request;
            new_request.first_byte  = job.byte_offset;
            new_request.last_byte   = job_end;
            new_request.job_indices.push_back(job_index);
            merged_requests.push_back(std::move(new_request));
        }
    }

    // Allocate per-request response buffers and curl contexts.
    size_t num_requests = merged_requests.size();
    std::vector<std::vector<uint8_t>> response_buffers(num_requests);
    std::vector<CurlWriteContext>     write_contexts(num_requests);
    std::vector<CURL*>                curl_handles(num_requests, nullptr);

    // Initialise the curl multi handle.
    CURLM* multi_handle = curl_multi_init();
    curl_multi_setopt(multi_handle, CURLMOPT_MAX_TOTAL_CONNECTIONS, 8L);

    // Build and enqueue one easy handle per merged request.
    for (size_t req_idx = 0; req_idx < num_requests; ++req_idx) {
        const MergedRangeRequest& req = merged_requests[req_idx];

        response_buffers[req_idx].reserve(req.last_byte - req.first_byte);
        write_contexts[req_idx].buffer = &response_buffers[req_idx];

        std::string range_value =
            "bytes=" + std::to_string(req.first_byte)
            + "-"    + std::to_string(req.last_byte - 1);

        std::string auth_header;
        std::string url = build_s3_request_url(location, range_value, auth_header);

        CURL* easy = curl_easy_init();
        curl_easy_setopt(easy, CURLOPT_URL,           url.c_str());
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_response_bytes);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA,     &write_contexts[req_idx]);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("Range: " + range_value).c_str());
        if (!location.is_anonymous) {
            headers = curl_slist_append(headers, ("Authorization: " + auth_header).c_str());
        }
        curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);

        curl_handles[req_idx] = easy;
        curl_multi_add_handle(multi_handle, easy);
    }

    // Run the event loop until all transfers finish.
    int transfers_running = 0;
    do {
        curl_multi_perform(multi_handle, &transfers_running);
        if (transfers_running > 0) {
            int wait_result = 0;
            curl_multi_wait(multi_handle, nullptr, 0, /*timeout_ms=*/50, &wait_result);
        }
    } while (transfers_running > 0);

    // Demultiplex response buffers back into individual job data.
    for (size_t req_idx = 0; req_idx < num_requests; ++req_idx) {
        const MergedRangeRequest&    req    = merged_requests[req_idx];
        const std::vector<uint8_t>&  buffer = response_buffers[req_idx];

        for (int job_index : req.job_indices) {
            TileFetch& job = jobs[job_index];

            size_t local_start = job.byte_offset - req.first_byte;
            size_t local_end   = local_start + job.byte_length;

            if (local_end <= buffer.size()) {
                job.data.assign(buffer.begin() + static_cast<long>(local_start),
                                buffer.begin() + static_cast<long>(local_end));
                job.err = 0;
            } else {
                job.err = 1;
            }
        }

        curl_multi_remove_handle(multi_handle, curl_handles[req_idx]);
        curl_easy_cleanup(curl_handles[req_idx]);
    }

    curl_multi_cleanup(multi_handle);
}
