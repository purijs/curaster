/**
 * @file chain.h
 * @brief Public Chain API — immutable lazy-evaluation raster pipeline.
 *
 * Each Chain wraps an input file path and an ordered list of operations.
 * Operations are applied lazily: no GPU work happens until one of the
 * terminal methods (save_local, save_s3, to_memory, iter_begin) is called.
 *
 * All builder methods (algebra, clip, reproject) return a *new* Chain so
 * the original remains unchanged — safe for parallel use.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../include/pipeline.h"  // ChainOp, ReprojectParams
#include "../../include/types.h"     // FileInfo, RasterResult
#include "../../include/chunk_queue.h"

// Forward declaration — avoids pulling in GDAL headers.
class GDALRasterBand;

/**
 * @brief Lazy, immutable raster processing pipeline.
 *
 * Create with curaster.open(path), then chain operations:
 * @code
 * chain.algebra("(B1 - B2) / (B1 + B2)")
 *      .reproject("EPSG:4326", 0.001, 0.001)
 *      .save_local("ndvi.tif")
 * @endcode
 */
class Chain {
public:
    /// Construct a Chain that reads from @p input_file.
    explicit Chain(const std::string& input_file);

    /// Copy constructor — used by builder methods to clone the chain.
    Chain(const Chain& other);

    // ── Builder methods (return a new Chain) ──────────────────────────────

    /**
     * @brief Append a band-math algebra operation.
     * @param expression  Expression string, e.g. "(B1 + B2) / 2".
     */
    std::shared_ptr<Chain> algebra(const std::string& expression);

    /**
     * @brief Append a polygon clip operation.
     * @param geojson  GeoJSON string (Polygon or MultiPolygon).
     */
    std::shared_ptr<Chain> clip(const std::string& geojson);

    /**
     * @brief Append a reprojection operation.
     * @param target_crs  Any CRS string GDAL understands (EPSG code, WKT, proj).
     * @param res_x       Output pixel width  (0 = auto).
     * @param res_y       Output pixel height (0 = auto).
     * @param resampling  "bilinear" (default) or "nearest".
     * @param nodata      Fill value for pixels outside the source extent.
     * @param te_xmin, te_ymin, te_xmax, te_ymax  Optional output extent in target CRS.
     */
    std::shared_ptr<Chain> reproject(const std::string& target_crs,
                                     double res_x       = 0,
                                     double res_y       = 0,
                                     const std::string& resampling = "bilinear",
                                     double nodata      = -9999.0,
                                     double te_xmin     = 0,
                                     double te_ymin     = 0,
                                     double te_xmax     = 0,
                                     double te_ymax     = 0);

    // ── Info ──────────────────────────────────────────────────────────────

    /**
     * @brief Return the FileInfo for the output raster (accounting for any REPROJECT op).
     *
     * Called by bindings.cpp to build the Python dict, and by terminal methods
     * internally.  Returns pure C++ types so chain.h has no pybind11 dependency.
     */
    FileInfo get_output_info() const;

    // ── Terminal methods ──────────────────────────────────────────────────

    /// Execute the pipeline and write results to a local GeoTIFF file.
    void save_local(const std::string& output_path, bool verbose = false);

    /// Execute the pipeline and upload results to an S3 object.
    void save_s3(const std::string& s3_path, bool verbose = false);

    /**
     * @brief Execute the pipeline and return all pixels in a RasterResult.
     * @throws std::runtime_error if the result would exceed 75% of available RAM.
     */
    std::shared_ptr<RasterResult> to_memory(bool verbose = false);

    /**
     * @brief Start background execution and return a ChunkQueue for streaming.
     *
     * The caller iterates by calling ChunkQueue::pop() until it returns false.
     * @param buffer_chunk_count  Maximum number of chunks to buffer (backpressure).
     */
    std::shared_ptr<ChunkQueue> iter_begin(int buffer_chunk_count = 4);

private:
    std::string            input_file_; ///< Path to the source GeoTIFF
    std::vector<ChainOp>   operations_; ///< Ordered list of lazy operations

    /// True if this chain contains a REPROJECT operation.
    bool has_reproject_operation() const;

    /**
     * @brief Internal: execute the full pipeline.
     *
     * @param output_band     If non-null, write each chunk via GDAL RasterIO.
     * @param result          If non-null, copy chunk data into result->data.
     * @param chunk_queue     If non-null, push each chunk onto the queue.
     * @param verbose         Print GDAL progress bar.
     */
    void execute(GDALRasterBand*                output_band,
                 RasterResult*                  result,
                 std::shared_ptr<ChunkQueue>    chunk_queue,
                 bool                           verbose = false);
};
