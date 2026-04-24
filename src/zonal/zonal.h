#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "../../include/pipeline.h"
#include "../../include/raster_core.h"
#include "../../include/types.h"








struct PrebuiltZone {
    std::vector<double> px;
    std::vector<double> py;
    uint16_t            id;
};

/// Parse a GeoJSON string into pixel-space vertex arrays.  Call once, then
/// pass the result to rasterize_zones_prebuilt() for each chunk.
std::vector<PrebuiltZone> build_prebuilt_zones(
    const std::string& geojson_str,
    const FileInfo&    file_info);

/// Rasterize prebuilt zones for a given chunk row range.
/// This avoids repeated GeoJSON parsing across chunks.
void rasterize_zones_prebuilt(
    const std::vector<PrebuiltZone>& zones,
    const FileInfo&                  file_info,
    int                              chunk_y0,
    int                              chunk_height,
    uint16_t*                        out_labels);


int rasterize_zones_chunked(
    const std::string& geojson_str,
    const FileInfo&    file_info,
    int                chunk_y0,
    int                chunk_height,
    uint16_t*          out_labels);



int count_zones_geojson(const std::string& geojson_str);


std::vector<ZoneResult> aggregate_zonal_results(
    const int*   h_count,
    const float* h_sum,
    const float* h_sum_sq,
    const float* h_min,
    const float* h_max,
    int          num_zones,
    const std::vector<std::string>& stats);
