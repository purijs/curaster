/**
 * @file clip.h
 * @brief Polygon scan-conversion: GeoJSON → per-row GPU span tables.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "../../include/raster_core.h"  
#include "../../include/types.h"        

/**
 * @brief Convert a GeoJSON polygon/multipolygon into per-row GPU span rows.
 *
 * Scan-converts polygon rings into horizontal [start, end] column spans
 * in image pixel coordinates.  The resulting table is used at execution
 * time by the GPU mask kernel to zero out pixels outside the polygon.
 *
 * @param geojson_str   GeoJSON string containing a Polygon or MultiPolygon.
 * @param file_info     Source raster metadata (width, height, geotransform).
 * @param spans_by_row  Output: keyed by image row, each entry holds a single
 *                      GpuSpanRow covering all spans for that row.
 * @throws std::runtime_error if the GeoJSON string is invalid.
 */
void parse_polygon_to_spans(
    const std::string&                              geojson_str,
    const FileInfo&                                 file_info,
    std::unordered_map<int, std::vector<GpuSpanRow>>& spans_by_row);
