/**
 * @file clip.cpp
 * @brief GeoJSON polygon scan-conversion into per-row GPU span tables.
 */
#include "clip.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <ogr_geometry.h>
#include <ogr_spatialref.h>
#include "gdal_priv.h"

// ─── Ring scan-conversion helper ─────────────────────────────────────────────

/**
 * @brief Scan-convert one OGR polygon ring into the spans_by_row table.
 *
 * Transforms ring vertices from geographic to image-pixel coordinates using
 * the inverse geotransform, then computes X-intercepts at each integer row
 * via edge-intersection and stores [left, right] column pairs.
 */
static void scan_convert_ring(
    OGRLinearRing*                                   ring,
    const double                                     inv_gt[6],
    const FileInfo&                                  file_info,
    std::unordered_map<int, std::vector<GpuSpanRow>>& spans_by_row) {

    int num_points = ring->getNumPoints();

    // Project all ring vertices from geographic to pixel space.
    std::vector<double> pixel_x(num_points);
    std::vector<double> pixel_y(num_points);

    for (int pt = 0; pt < num_points; ++pt) {
        double geo_x = ring->getX(pt);
        double geo_y = ring->getY(pt);
        pixel_x[pt] = inv_gt[0] + geo_x * inv_gt[1] + geo_y * inv_gt[2];
        pixel_y[pt] = inv_gt[3] + geo_x * inv_gt[4] + geo_y * inv_gt[5];
    }

    // Clamp row range to the raster extent.
    int row_min = static_cast<int>(
        std::max(0.0, *std::min_element(pixel_y.begin(), pixel_y.end())));
    int row_max = static_cast<int>(
        std::min(static_cast<double>(file_info.height - 1),
                 *std::max_element(pixel_y.begin(), pixel_y.end())));

    // For each scanline, find the X-intercepts of all polygon edges.
    for (int row = row_min; row <= row_max; ++row) {
        std::vector<double> x_intercepts;

        for (int edge = 0; edge < num_points - 1; ++edge) {
            double edge_y0 = pixel_y[edge];
            double edge_y1 = pixel_y[edge + 1];
            double edge_x0 = pixel_x[edge];
            double edge_x1 = pixel_x[edge + 1];

            // Only count edges that cross this scanline.
            bool crosses_row =
                (edge_y0 <= row && row < edge_y1) ||
                (edge_y1 <= row && row < edge_y0);

            if (crosses_row) {
                double x_at_row = edge_x0 + (row - edge_y0) / (edge_y1 - edge_y0)
                                           * (edge_x1 - edge_x0);
                x_intercepts.push_back(x_at_row);
            }
        }

        std::sort(x_intercepts.begin(), x_intercepts.end());

        // Pair up intercepts to form [left, right] spans.
        if (x_intercepts.size() >= 2) {
            auto& row_entries = spans_by_row[row];
            if (row_entries.empty()) {
                row_entries.resize(1);
            }
            GpuSpanRow& span_row = row_entries[0];

            if (span_row.num_spans < 1024) {
                int col_left  = std::max(0,                   static_cast<int>(x_intercepts[0]));
                int col_right = std::min(file_info.width - 1, static_cast<int>(x_intercepts[1]));
                span_row.spans[span_row.num_spans++] = {col_left, col_right};
            }
        }
    }
}

// ─── parse_polygon_to_spans ──────────────────────────────────────────────────

void parse_polygon_to_spans(
    const std::string&                               geojson_str,
    const FileInfo&                                  file_info,
    std::unordered_map<int, std::vector<GpuSpanRow>>& spans_by_row) {

    OGRGeometry* geometry = OGRGeometryFactory::createFromGeoJson(geojson_str.c_str());
    if (!geometry) {
        throw std::runtime_error("Invalid GeoJSON for clip operation.");
    }

    // Compute the inverse geotransform for geo→pixel projection.
    double inv_gt[6];
    GDALInvGeoTransform(const_cast<double*>(file_info.geo_transform), inv_gt);

    OGRwkbGeometryType geom_type = wkbFlatten(geometry->getGeometryType());

    if (geom_type == wkbPolygon) {
        auto* polygon = static_cast<OGRPolygon*>(geometry);
        scan_convert_ring(polygon->getExteriorRing(), inv_gt, file_info, spans_by_row);

    } else if (geom_type == wkbMultiPolygon) {
        auto* multi_polygon = static_cast<OGRMultiPolygon*>(geometry);
        int   num_geometries = multi_polygon->getNumGeometries();

        for (int geom_idx = 0; geom_idx < num_geometries; ++geom_idx) {
            auto* sub_polygon = static_cast<OGRPolygon*>(
                multi_polygon->getGeometryRef(geom_idx));
            scan_convert_ring(sub_polygon->getExteriorRing(), inv_gt, file_info, spans_by_row);
        }
    }

    OGRGeometryFactory::destroyGeometry(geometry);
}
