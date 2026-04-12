/**
 * @file reproject.h
 * @brief Coordinate reprojection helpers: output extent calculation,
 *        coarse grid computation, and bbox derivation.
 *
 * The reprojection pipeline works in two phases:
 *  1. Pre-pass (pre_pass_reproject): compute the output FileInfo (extent,
 *     resolution, WKT CRS) from source metadata and user parameters.
 *  2. Per-chunk (WarpTransformer + compute_coarse_grid + coarse_to_bbox):
 *     for each destination chunk, build the 32×32 control-point grid that
 *     maps destination pixels to source canvas coordinates.
 */
#pragma once

#include "../../include/types.h"     // FileInfo, SrcBBox

struct ReprojectParams; // Defined in pipeline.h

// ─── Pre-pass: compute output FileInfo ──────────────────────────────────────

/**
 * @brief Determine the output raster extent and resolution for a reprojection.
 *
 * Projects the four corners of the source raster into the target CRS,
 * derives a pixel size, and constructs a FileInfo describing the output grid.
 *
 * @param input_file Path to the source file (used only to open with GDAL if needed).
 * @param src_info   Source raster metadata.
 * @param params     Reprojection parameters (target CRS, resolution, optional extent).
 * @return           FileInfo for the output raster in the target CRS.
 */
FileInfo pre_pass_reproject(const std::string&     input_file,
                            const FileInfo&        src_info,
                            const ReprojectParams& params);

// ─── Per-chunk coordinate transformer ────────────────────────────────────────

/**
 * @brief Encapsulates the OGR coordinate transformation for one worker thread.
 *
 * Transforms (destination pixel) → (source pixel) so the warp kernel can
 * sample the source canvas at the correct location for each output pixel.
 */
class WarpTransformer {
public:
    /**
     * @brief Initialise the transformer for src → dst coordinate mapping.
     *
     * Creates an OGR coordinate transformation (dst CRS → src CRS) and
     * stores the geotransforms needed to convert between pixel and geographic space.
     */
    void initialise(const FileInfo& src_info, const FileInfo& dst_info);

    /**
     * @brief Transform @p count (x, y) destination-pixel coordinates in-place
     *        to source-pixel coordinates.
     *
     * Applies: dst_pixel → dst_geo → src_geo → src_pixel  in the three
     * steps required by the (inverse) reprojection pipeline.
     */
    void transform_pixels(double* x_coords, double* y_coords, int count) const;

    /// Release the underlying OGR coordinate transformation object.
    void destroy();

private:
    void*  ogr_transform_ = nullptr;  ///< OGRCoordinateTransformation* (held as void* to avoid GDAL in header)
    double src_inv_geotransform_[6];  ///< Inverse geotransform of the source raster
    double dst_geotransform_[6];      ///< Forward geotransform of the destination raster
};

// ─── Per-chunk coarse grid ────────────────────────────────────────────────────

/**
 * @brief Fill the 32×32 coarse warp grid with source-pixel coordinates.
 *
 * Evenly spaces WARP_GRID_WIDTH × WARP_GRID_HEIGHT control points across the
 * destination chunk, transforms each to source-pixel coordinates, and writes
 * the results into @p out_x and @p out_y flat arrays.
 *
 * @param transformer     Initialised transformer for the current chunk.
 * @param chunk_y0_dst    Top row of the destination chunk in image coordinates.
 * @param chunk_height_dst Height of the destination chunk.
 * @param dst_width       Width of the destination raster.
 * @param out_x           Output: [WARP_GRID_HEIGHT × WARP_GRID_WIDTH] source X coords.
 * @param out_y           Output: [WARP_GRID_HEIGHT × WARP_GRID_WIDTH] source Y coords.
 */
void compute_coarse_grid(const WarpTransformer& transformer,
                         int                    chunk_y0_dst,
                         int                    chunk_height_dst,
                         int                    dst_width,
                         double*                out_x,
                         double*                out_y);

/**
 * @brief Derive a source bounding box from the coarse grid control points.
 *
 * Adds a 2-pixel margin around the min/max of the grid coordinates and
 * clamps to the source raster extent.
 *
 * @param grid_x   Source X coordinates of the control points.
 * @param grid_y   Source Y coordinates of the control points.
 * @param src_width  Source raster width  (for clamping).
 * @param src_height Source raster height (for clamping).
 * @return         Bounding box in source-pixel coordinates.
 */
SrcBBox coarse_grid_to_source_bbox(const double* grid_x, const double* grid_y,
                                   int src_width, int src_height);
