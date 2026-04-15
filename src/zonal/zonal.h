#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "../../include/pipeline.h"
#include "../../include/raster_core.h"
#include "../../include/types.h"

// Count the number of distinct polygon zones in a GeoJSON string.
int count_zones_geojson(const std::string& geojson_str);

// Rasterize multiple GeoJSON polygons into a per-chunk uint16_t zone label array.
// zone_id == 0 means "no zone". Returns number of zones found.
int rasterize_zones_chunked(
    const std::string& geojson_str,
    const FileInfo&    file_info,
    int                chunk_y0,
    int                chunk_height,
    uint16_t*          out_labels);


// Aggregate device-side per-zone accumulators into ZoneResult list.
std::vector<ZoneResult> aggregate_zonal_results(
    const int*   h_count,
    const float* h_sum,
    const float* h_sum_sq,
    const float* h_min,
    const float* h_max,
    int          num_zones,
    const std::vector<std::string>& stats);
