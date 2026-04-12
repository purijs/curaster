/**
 * @file s3_fetch.h
 * @brief S3 tile / strip fetcher using libcurl multi.
 *
 * Merges adjacent byte ranges to minimise the number of HTTP requests,
 * then issues them in parallel via a CURLM handle and demultiplexes
 * the data back into the individual TileFetch jobs.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "s3_auth.h"  // S3Loc

// ─── Tile fetch job ───────────────────────────────────────────────────────────
/**
 * @brief Describes one tile or strip to fetch from S3.
 *
 * The caller fills tile_index, byte_offset, and byte_length.
 * s3_fetch_tiles() fills data and sets err on completion.
 */
struct TileFetch {
    size_t               tile_index;   ///< Index into tile_offsets / tile_lengths (for demux)
    size_t               byte_offset;  ///< Absolute byte offset of the tile in the file
    size_t               byte_length;  ///< Compressed byte length to fetch
    std::vector<uint8_t> data;         ///< Filled by s3_fetch_tiles() on success
    int                  err;          ///< 0 = success, 1 = fetch or demux error
};

// ─── Parallel fetch ───────────────────────────────────────────────────────────
/**
 * @brief Fetch all tiles described in @p jobs from @p location in parallel.
 *
 * Adjacent requests within 128 KB of each other are merged into a single
 * HTTP Range request to reduce round-trips.  Up to 8 requests are issued
 * concurrently via libcurl multi.
 *
 * @param location      Authenticated S3 location.
 * @param all_offsets   Full tile-offset table (used to build Range headers).
 * @param all_lengths   Full tile-length table.
 * @param jobs          In/out: jobs to fetch; data and err are filled in.
 */
void s3_fetch_tiles(const S3Loc&              location,
                    const std::vector<size_t>& all_offsets,
                    const std::vector<size_t>& all_lengths,
                    std::vector<TileFetch>&    jobs);
