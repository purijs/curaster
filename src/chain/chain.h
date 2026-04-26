/**
 * @file chain.h
 * @brief Public Chain API — immutable lazy-evaluation raster pipeline.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../include/pipeline.h"
#include "../../include/types.h"
#include "../../include/chunk_queue.h"

// Full VramCache definition is in vram_cache.h; forward decl is enough here
// because Chain only stores a shared_ptr<VramCache>.
struct VramCache;

class GDALRasterBand;


class Chain {
public:
    explicit Chain(const std::string& input_file);
    Chain(const Chain& other);


    

    /**
     * @brief Pre-load the source raster into VRAM for zero-I/O repeated processing.
     *
     * Decodes all bands into CUDA device memory and creates bilinear + nearest
     * texture objects for the warp engine.  All chains derived from the returned
     * chain (via algebra/clip/reproject/…) share the same VRAM cache via
     * shared_ptr, so each band is stored in VRAM only once.
     *
     * Operations and outputs are completely unchanged; only the I/O path is
     * replaced: instead of reading from disk or S3 on every operation, the
     * engines read directly from VRAM.
     *
     * @throws std::runtime_error if the decoded raster (data + texture arrays)
     *         would exceed 80% of currently-free VRAM, or if raster dimensions
     *         exceed the CUDA 2D texture limit on the active GPU.
     */
    std::shared_ptr<Chain> persist();

    std::shared_ptr<Chain> select_bands(std::vector<int> bands);

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

    std::shared_ptr<Chain> focal(const std::string& stat = "mean",
                                  int radius             = 1,
                                  const std::string& shape = "square",
                                  bool clamp_border      = true);

    std::shared_ptr<Chain> terrain(const std::vector<std::string>& metrics = {},
                                    const std::string& unit     = "degrees",
                                    double sun_azimuth          = 315.0,
                                    double sun_altitude         = 45.0,
                                    const std::string& method   = "horn");

    std::shared_ptr<Chain> texture(const std::vector<std::string>& features = {},
                                    int window                 = 11,
                                    int levels                 = 32,
                                    const std::string& direction_mode = "average",
                                    bool log_scale             = false,
                                    float val_min              = 0.f,
                                    float val_max              = 0.f);

    std::vector<ZoneResult> zonal_stats(const std::vector<std::string>& stats = {},
                                         int band = 1,
                                         const std::string& geojson_str = "",
                                         bool verbose = false);


    /**
     * @brief Return the FileInfo for the output raster (accounting for any REPROJECT op).
     *
     * Called by bindings.cpp to build the Python dict, and by terminal methods
     * internally.  Returns pure C++ types so chain.h has no pybind11 dependency.
     */
    FileInfo get_output_info() const;

    

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
    std::string            input_file_;
    std::vector<ChainOp>   operations_;

    /// Non-null when persist() was called on an ancestor chain.
    /// Shared across all derived chains; freed when the last chain is destroyed.
    std::shared_ptr<VramCache> cache_;
    std::vector<int> user_band_selection_;

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
