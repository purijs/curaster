/**
 * @file reproject.cpp
 * @brief Pre-pass extent calculation, WarpTransformer, and coarse-grid routines.
 */
#include "reproject.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "gdal_priv.h"
#include "ogr_spatialref.h"
#include "../../include/pipeline.h"   // ReprojectParams
#include "../../include/raster_core.h" // WARP_GRID_WIDTH, WARP_GRID_HEIGHT

// ─── pre_pass_reproject ───────────────────────────────────────────────────────

FileInfo pre_pass_reproject(const std::string&     /*input_file*/,
                            const FileInfo&        src_info,
                            const ReprojectParams& params) {

    // Build the destination SRS from whatever string GDAL understands.
    OGRSpatialReference dst_srs;
#if GDAL_VERSION_MAJOR >= 3
    dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif
    if (dst_srs.SetFromUserInput(params.target_crs.c_str()) != OGRERR_NONE) {
        throw std::runtime_error("Invalid target CRS: " + params.target_crs);
    }

    char* dst_wkt_raw = nullptr;
    dst_srs.exportToWkt(&dst_wkt_raw);
    std::string dst_wkt(dst_wkt_raw);
    CPLFree(dst_wkt_raw);

    // Build the source SRS from the source file's WKT.
    OGRSpatialReference src_srs;
#if GDAL_VERSION_MAJOR >= 3
    src_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif
    src_srs.importFromWkt(src_info.projection.c_str());

    // Create a forward coordinate transformation (src → dst) to project corners.
    OGRCoordinateTransformation* forward_ct =
        OGRCreateCoordinateTransformation(&src_srs, &dst_srs);
    if (!forward_ct) {
        throw std::runtime_error("Cannot create forward coordinate transformation.");
    }

    // Project the four corners of the source raster into the target CRS.
    const double W = static_cast<double>(src_info.width);
    const double H = static_cast<double>(src_info.height);
    const double* gt = src_info.geo_transform;

    double corner_x[4] = { gt[0],          gt[0] + W*gt[1], gt[0] + H*gt[2], gt[0] + W*gt[1] + H*gt[2] };
    double corner_y[4] = { gt[3],          gt[3] + W*gt[4], gt[3] + H*gt[5], gt[3] + W*gt[4] + H*gt[5] };
    forward_ct->Transform(4, corner_x, corner_y);
    OGRCoordinateTransformation::DestroyCT(forward_ct);

    double projected_xmin = *std::min_element(corner_x, corner_x + 4);
    double projected_xmax = *std::max_element(corner_x, corner_x + 4);
    double projected_ymin = *std::min_element(corner_y, corner_y + 4);
    double projected_ymax = *std::max_element(corner_y, corner_y + 4);

    // Override with user-specified extent if provided.
    if (params.has_extent) {
        projected_xmin = params.extent_xmin;
        projected_xmax = params.extent_xmax;
        projected_ymin = params.extent_ymin;
        projected_ymax = params.extent_ymax;
    }

    // Derive pixel size: use user-specified values or auto-compute from source.
    double pixel_size_x = params.pixel_size_x;
    double pixel_size_y = params.pixel_size_y;
    if (pixel_size_x <= 0 || pixel_size_y <= 0) {
        pixel_size_x = (projected_xmax - projected_xmin) / src_info.width;
        pixel_size_y = (projected_ymax - projected_ymin) / src_info.height;
    }

    // Build the output FileInfo.
    FileInfo out_info;
    out_info.width  = std::max(1, static_cast<int>(
        std::round((projected_xmax - projected_xmin) / pixel_size_x)));
    out_info.height = std::max(1, static_cast<int>(
        std::round((projected_ymax - projected_ymin) / pixel_size_y)));

    out_info.geo_transform[0] = projected_xmin;
    out_info.geo_transform[1] = pixel_size_x;
    out_info.geo_transform[2] = 0.0;
    out_info.geo_transform[3] = projected_ymax;
    out_info.geo_transform[4] = 0.0;
    out_info.geo_transform[5] = -pixel_size_y;

    out_info.projection       = dst_wkt;
    out_info.tile_width       = 512;
    out_info.tile_height      = 512;
    out_info.data_type        = 6;  // GDT_Float32
    out_info.samples_per_pixel = src_info.samples_per_pixel;
    out_info.interleave        = "BAND";
    out_info.is_tiled          = true;

    return out_info;
}

// ─── WarpTransformer ──────────────────────────────────────────────────────────

void WarpTransformer::initialise(const FileInfo& src_info, const FileInfo& dst_info) {
    OGRSpatialReference src_srs, dst_srs;
#if GDAL_VERSION_MAJOR >= 3
    src_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    dst_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif
    src_srs.importFromWkt(src_info.projection.c_str());
    dst_srs.importFromWkt(dst_info.projection.c_str());

    // Note: direction is dst → src (inverse warp).
    auto* ct = OGRCreateCoordinateTransformation(&dst_srs, &src_srs);
    if (!ct) {
        throw std::runtime_error("Cannot create coordinate transformation for warp.");
    }
    ogr_transform_ = ct;

    double src_gt_copy[6];
    memcpy(src_gt_copy, src_info.geo_transform, sizeof(src_gt_copy));
    GDALInvGeoTransform(src_gt_copy, src_inv_geotransform_);
    memcpy(dst_geotransform_, dst_info.geo_transform, sizeof(dst_geotransform_));
}

void WarpTransformer::transform_pixels(double* x_coords, double* y_coords, int count) const {
    // Step 1: destination pixel → destination geographic.
    for (int i = 0; i < count; ++i) {
        double dst_col = x_coords[i];
        double dst_row = y_coords[i];
        x_coords[i] = dst_geotransform_[0] + dst_col * dst_geotransform_[1]
                                            + dst_row * dst_geotransform_[2];
        y_coords[i] = dst_geotransform_[3] + dst_col * dst_geotransform_[4]
                                            + dst_row * dst_geotransform_[5];
    }

    // Step 2: destination geographic → source geographic (OGR).
    static_cast<OGRCoordinateTransformation*>(ogr_transform_)
        ->Transform(count, x_coords, y_coords);

    // Step 3: source geographic → source pixel.
    for (int i = 0; i < count; ++i) {
        double src_geo_x = x_coords[i];
        double src_geo_y = y_coords[i];
        x_coords[i] = src_inv_geotransform_[0] + src_geo_x * src_inv_geotransform_[1]
                                                + src_geo_y * src_inv_geotransform_[2];
        y_coords[i] = src_inv_geotransform_[3] + src_geo_x * src_inv_geotransform_[4]
                                                + src_geo_y * src_inv_geotransform_[5];
    }
}

void WarpTransformer::destroy() {
    if (ogr_transform_) {
        OGRCoordinateTransformation::DestroyCT(
            static_cast<OGRCoordinateTransformation*>(ogr_transform_));
        ogr_transform_ = nullptr;
    }
}

// ─── compute_coarse_grid ─────────────────────────────────────────────────────

void compute_coarse_grid(const WarpTransformer& transformer,
                         int                    chunk_y0_dst,
                         int                    chunk_height_dst,
                         int                    dst_width,
                         double*                out_x,
                         double*                out_y) {

    // Fill the grid with evenly-spaced destination pixel coordinates.
    for (int grid_row = 0; grid_row < WARP_GRID_HEIGHT; ++grid_row) {
        for (int grid_col = 0; grid_col < WARP_GRID_WIDTH; ++grid_col) {

            double dst_col = (WARP_GRID_WIDTH > 1)
                ? static_cast<double>(grid_col) / (WARP_GRID_WIDTH  - 1) * (dst_width       - 1)
                : 0.0;
            double dst_row = (WARP_GRID_HEIGHT > 1)
                ? chunk_y0_dst + static_cast<double>(grid_row) / (WARP_GRID_HEIGHT - 1)
                                                                * (chunk_height_dst - 1)
                : static_cast<double>(chunk_y0_dst);

            out_x[grid_row * WARP_GRID_WIDTH + grid_col] = dst_col;
            out_y[grid_row * WARP_GRID_WIDTH + grid_col] = dst_row;
        }
    }

    // Transform all control points from destination pixel → source pixel in one call.
    transformer.transform_pixels(out_x, out_y, WARP_GRID_WIDTH * WARP_GRID_HEIGHT);
}

// ─── coarse_grid_to_source_bbox ──────────────────────────────────────────────

SrcBBox coarse_grid_to_source_bbox(const double* grid_x, const double* grid_y,
                                   int src_width, int src_height) {
    const int num_control_points = WARP_GRID_WIDTH * WARP_GRID_HEIGHT;

    double x_min = grid_x[0], x_max = grid_x[0];
    double y_min = grid_y[0], y_max = grid_y[0];

    for (int i = 1; i < num_control_points; ++i) {
        x_min = std::min(x_min, grid_x[i]);
        x_max = std::max(x_max, grid_x[i]);
        y_min = std::min(y_min, grid_y[i]);
        y_max = std::max(y_max, grid_y[i]);
    }

    // Add a 2-pixel margin and clamp to the source raster.
    SrcBBox bbox;
    bbox.x0 = std::max(0,            static_cast<int>(std::floor(x_min)) - 2);
    bbox.y0 = std::max(0,            static_cast<int>(std::floor(y_min)) - 2);
    int x1  = std::min(src_width  - 1, static_cast<int>(std::ceil(x_max)) + 2);
    int y1  = std::min(src_height - 1, static_cast<int>(std::ceil(y_max)) + 2);
    bbox.w  = std::max(1, x1 - bbox.x0 + 1);
    bbox.h  = std::max(1, y1 - bbox.y0 + 1);

    return bbox;
}
