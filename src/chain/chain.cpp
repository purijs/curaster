#include "chain.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gdal_priv.h"
#include "cpl_vsi.h"
#include "cpl_conv.h"

#include "../../include/pipeline.h"
#include "../../include/types.h"
#include "../../include/chunk_queue.h"

#include "../tiff/tiff_metadata.h"
#include "../algebra/algebra_compiler.h"
#include "../clip/clip.h"
#include "../focal/focal.h"
#include "../texture/texture.h"
#include "../engine/engine.h"
#include "../engine/engine_focal.h"
#include "../engine/engine_zonal.h"
#include "../reproject/reproject.h"



Chain::Chain(const std::string& input_file)
    : input_file_(input_file) {}

Chain::Chain(const Chain& other)
    : input_file_(other.input_file_)
    , operations_(other.operations_) {}



bool Chain::has_reproject_operation() const {
    for (const auto& op : operations_) {
        if (op.type == ChainOpType::REPROJECT) { return true; }
    }
    return false;
}

FileInfo Chain::get_output_info() const {
    FileInfo src_info = get_file_info(input_file_);
    for (const auto& op : operations_) {
        if (op.type == ChainOpType::REPROJECT) {
            return pre_pass_reproject(input_file_, src_info, op.reproject_params);
        }
    }
    return src_info;
}



static PipelineCtx build_pipeline_context(const std::vector<ChainOp>& operations,
                                           const FileInfo&              src_info,
                                           const std::string&           input_file) {
    PipelineCtx ctx;

    for (const auto& op : operations) {
        if (op.type == ChainOpType::REPROJECT) {
            ctx.has_reproject        = true;
            ctx.reproject_params     = op.reproject_params;
            ctx.reproject_output_info = pre_pass_reproject(input_file, src_info,
                                                            op.reproject_params);
            break;
        }
    }

    const FileInfo& clip_reference_info = ctx.has_reproject
                                        ? ctx.reproject_output_info
                                        : src_info;

    std::vector<int> band_map;
    bool has_algebra       = false;
    bool has_neighborhood  = false;

    for (const auto& op : operations) {
        if (op.type == ChainOpType::ALGEBRA) {
            ctx.instructions = compile_algebra_expression(op.algebra_expr, band_map);
            has_algebra = true;
        }
        if (op.type == ChainOpType::CLIP) {
            ctx.has_clip_mask = true;
            parse_polygon_to_spans(op.geojson_str, clip_reference_info, ctx.clip_spans);
        }
        if (op.type == ChainOpType::FOCAL) {
            ctx.has_focal   = true;
            ctx.focal_params = op.focal_params;
            ctx.focal_num_output_bands = 1;
            has_neighborhood = true;
        }
        if (op.type == ChainOpType::TERRAIN) {
            ctx.has_terrain    = true;
            ctx.terrain_params = op.terrain_params;
            ctx.focal_num_output_bands = op.terrain_params.num_output_bands;
            has_neighborhood = true;
        }
        if (op.type == ChainOpType::TEXTURE) {
            ctx.has_texture  = true;
            ctx.glcm_params  = op.glcm_params;
            ctx.focal_num_output_bands = op.glcm_params.num_output_bands;
            has_neighborhood = true;
        }
        if (op.type == ChainOpType::ZONAL_STATS) {
            ctx.has_zonal    = true;
            ctx.zonal_params = op.zonal_params;
        }
    }

    if (!has_algebra && !has_neighborhood && !ctx.has_zonal) {
        throw std::runtime_error("Chain must contain at least one algebra(), focal(), terrain(), texture(), or zonal_stats() operation.");
    }

    if (band_map.empty()) {
        band_map.push_back(0);
    }

    ctx.band_map = band_map;
    return ctx;
}



void Chain::execute(GDALRasterBand*             output_band,
                    RasterResult*               result,
                    std::shared_ptr<ChunkQueue> chunk_queue,
                    bool                        verbose) {
    FileInfo   src_info = get_file_info(input_file_);
    PipelineCtx ctx     = build_pipeline_context(operations_, src_info, input_file_);
    const FileInfo& out_info = ctx.has_reproject ? ctx.reproject_output_info : src_info;
    (void)out_info; 

    int num_out_bands = ctx.focal_num_output_bands;
    if (!ctx.has_focal && !ctx.has_terrain && !ctx.has_texture) num_out_bands = 1;

    if (output_band) {
        ctx.output_band = output_band;
    }
    if (result) {
        int total_width  = result->width;
        int total_height = result->height;
        int bands        = result->bands;
        ctx.result_callback = [result, total_width, total_height, bands]
                              (int width, int height, float* pixels, int y_offset) {
            size_t chunk_pixels = static_cast<size_t>(width) * height;
            size_t total_pixels = static_cast<size_t>(total_width) * total_height;
            for (int b = 0; b < bands; ++b) {
                memcpy(result->data.data() + b * total_pixels
                                           + static_cast<size_t>(y_offset) * width,
                       pixels + b * chunk_pixels,
                       chunk_pixels * sizeof(float));
            }
        };
    }
    if (chunk_queue) {
        ctx.queue_callback = [chunk_queue, num_out_bands](int width, int height, float* pixels, int y_offset) {
            ChunkResult chunk;
            chunk.width    = width;
            chunk.height   = height;
            chunk.y_offset = y_offset;
            chunk.data.assign(pixels, pixels + static_cast<size_t>(width) * height * num_out_bands);
            chunk_queue->push(std::move(chunk));
        };
    }

    if (ctx.has_focal || ctx.has_terrain || ctx.has_texture) {
        run_engine_focal(input_file_, ctx, verbose);
    } else if (ctx.has_zonal) {
        run_engine_zonal(input_file_, ctx, verbose);
    } else {
        run_engine_ex(input_file_, ctx, verbose);
    }
}

std::shared_ptr<Chain> Chain::algebra(const std::string& expression) {
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::ALGEBRA;
    op.algebra_expr = expression;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::clip(const std::string& geojson) {
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::CLIP;
    op.geojson_str = geojson;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::reproject(const std::string& target_crs,
                                         double             res_x,
                                         double             res_y,
                                         const std::string& resampling,
                                         double             nodata,
                                         double             te_xmin,
                                         double             te_ymin,
                                         double             te_xmax,
                                         double             te_ymax) {
    ReprojectParams rp;
    rp.target_crs   = target_crs;
    rp.pixel_size_x = res_x;
    rp.pixel_size_y = res_y;
    rp.resampling   = (resampling == "nearest")
                        ? ResampleMethod::NEAREST
                        : ResampleMethod::BILINEAR;
    rp.nodata_value = nodata;

    if (te_xmin != 0.0 || te_xmax != 0.0) {
        rp.has_extent  = true;
        rp.extent_xmin = te_xmin;
        rp.extent_ymin = te_ymin;
        rp.extent_xmax = te_xmax;
        rp.extent_ymax = te_ymax;
    }

    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::REPROJECT;
    op.reproject_params = rp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::focal(const std::string& stat,
                                     int radius,
                                     const std::string& shape,
                                     bool clamp_border)
{
    FocalParams fp;
    build_focal_params(fp, stat, radius, shape, clamp_border);
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::FOCAL;
    op.focal_params = fp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::terrain(const std::vector<std::string>& metrics,
                                       const std::string& unit,
                                       double sun_azimuth,
                                       double sun_altitude,
                                       const std::string& method)
{
    FileInfo src_info = get_file_info(input_file_);
    TerrainParams tp;
    build_terrain_params(tp, metrics.empty() ? std::vector<std::string>{"slope"} : metrics,
                         unit, sun_azimuth, sun_altitude, method, src_info);
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::TERRAIN;
    op.terrain_params = tp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::shared_ptr<Chain> Chain::texture(const std::vector<std::string>& features,
                                       int window,
                                       int levels,
                                       const std::string& direction_mode,
                                       bool log_scale,
                                       float val_min,
                                       float val_max)
{
    GLCMParams gp;
    build_glcm_params(gp, features, window, levels, direction_mode, log_scale, val_min, val_max);
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::TEXTURE;
    op.glcm_params = gp;
    new_chain->operations_.push_back(op);
    return new_chain;
}

std::vector<ZoneResult> Chain::zonal_stats(const std::string& geojson_str,
                                             const std::vector<std::string>& stats,
                                             int band,
                                             bool verbose)
{
    auto new_chain = std::make_shared<Chain>(*this);
    ChainOp op;
    op.type = ChainOpType::ZONAL_STATS;
    op.zonal_params.geojson_str = geojson_str;
    op.zonal_params.stats       = stats;
    op.zonal_params.band        = band;
    new_chain->operations_.push_back(op);

    FileInfo    src_info = get_file_info(new_chain->input_file_);
    PipelineCtx ctx      = build_pipeline_context(new_chain->operations_, src_info,
                                                   new_chain->input_file_);
    run_engine_zonal(new_chain->input_file_, ctx, verbose);
    return ctx.zonal_results;
}




void Chain::save_local(const std::string& output_path, bool verbose) {
    FileInfo src_info  = get_file_info(input_file_);
    PipelineCtx ctx    = build_pipeline_context(operations_, src_info, input_file_);
    int num_out_bands  = ctx.focal_num_output_bands;
    if (!ctx.has_focal && !ctx.has_terrain && !ctx.has_texture) num_out_bands = 1;

    FileInfo out_info  = get_output_info();
    GDALDataset* output_ds = create_output_dataset(output_path, out_info, num_out_bands);

    if (num_out_bands == 1) {
        GDALRasterBand* out_band = output_ds->GetRasterBand(1);
        execute(out_band, nullptr, nullptr, verbose);
    } else {
        ctx.output_dataset = static_cast<void*>(output_ds);
        run_engine_focal(input_file_, ctx, verbose);
    }
    GDALClose(output_ds);
}

void Chain::save_s3(const std::string& s3_path, bool verbose) {
    // Write to a real disk temp file first (chunk-by-chunk, no RAM accumulation),
    // then upload to S3 as a single sequential copy.  This avoids GDAL buffering
    // the whole output in /vsimem/ when CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES.
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    std::string tmp_path = std::string(tmp_dir) + "curaster_s3_"
                         + std::to_string(GetCurrentProcessId()) + ".tif";
#else
    std::string tmp_path = std::string("/tmp/curaster_s3_") + std::to_string(getpid()) + ".tif";
#endif

    try {
        save_local(tmp_path, verbose);
    } catch (...) {
        VSIUnlink(tmp_path.c_str());
        throw;
    }

    if (verbose) {
        printf("\n[save_s3] Uploading %s → %s\n", tmp_path.c_str(), s3_path.c_str());
        fflush(stdout);
    }

    if (CPLCopyFile(s3_path.c_str(), tmp_path.c_str()) != 0) {
        VSIUnlink(tmp_path.c_str());
        throw std::runtime_error(
            "S3 upload failed. Ensure the path starts with /vsis3/ and AWS credentials are valid.");
    }
    VSIUnlink(tmp_path.c_str());
}

std::shared_ptr<RasterResult> Chain::to_memory(bool verbose) {
    FileInfo src_info  = get_file_info(input_file_);
    PipelineCtx tmp_ctx = build_pipeline_context(operations_, src_info, input_file_);
    int num_out_bands  = tmp_ctx.focal_num_output_bands;
    if (!tmp_ctx.has_focal && !tmp_ctx.has_terrain && !tmp_ctx.has_texture) num_out_bands = 1;

    FileInfo out_info = get_output_info();

    size_t required_bytes = static_cast<size_t>(out_info.width)
                          * static_cast<size_t>(out_info.height)
                          * num_out_bands * sizeof(float);

    // Dynamic RAM budget: engine_focal/terrain/texture pre-claims 90% of available RAM
    // for its pinned halo pool (init_ram_budget). The output array must fit in the
    // remaining headroom. Plain algebra/clip/reproject pipelines can use most of RAM.
    double ram_fraction;
    if (tmp_ctx.has_focal || tmp_ctx.has_terrain || tmp_ctx.has_texture) {
        // engine_focal claims 0.90 * available_ram → only 0.10 left for output
        // Use 0.08 to stay safely below that ceiling.
        ram_fraction = 0.08;
    } else if (tmp_ctx.has_temporal) {
        // Temporal loads N full-scene buffers simultaneously.
        ram_fraction = 0.25;
    } else {
        // Simple algebra / clip / reproject — very little internal overhead.
        ram_fraction = 0.70;
    }

    size_t safe_ram_limit = static_cast<size_t>(get_available_ram() * ram_fraction);

    if (required_bytes > safe_ram_limit) {
        throw std::runtime_error(
            "MemoryError: raster requires " + std::to_string(required_bytes / 1048576)
            + " MB but only " + std::to_string(safe_ram_limit / 1048576)
            + " MB is safely available for output allocation (pipeline mode: "
            + (tmp_ctx.has_focal || tmp_ctx.has_terrain || tmp_ctx.has_texture
                ? "focal/terrain/texture" : tmp_ctx.has_temporal ? "temporal" : "standard")
            + "). Use iter_begin() to stream chunks instead.");
    }

    auto result      = std::make_shared<RasterResult>();
    result->width    = out_info.width;
    result->height   = out_info.height;
    result->bands    = num_out_bands;
    result->file_info   = out_info;
    result->projection  = out_info.projection;
    memcpy(result->geo_transform, out_info.geo_transform, sizeof(out_info.geo_transform));

    try {
        result->allocate();
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "MemoryError: raster requires " +
            std::to_string(required_bytes / 1048576) +
            " MB but allocation failed. Use iter_begin() to stream chunks instead.");
    }

    execute(nullptr, result.get(), nullptr, verbose);
    return result;
}

std::shared_ptr<ChunkQueue> Chain::iter_begin(int buffer_chunk_count) {
    auto queue = std::make_shared<ChunkQueue>(buffer_chunk_count);
    Chain chain_copy(*this);

    std::thread([chain_copy, queue]() mutable {
        try {
            chain_copy.execute(nullptr, nullptr, queue);
        } catch (...) {
            
        }
        queue->finish();
    }).detach();

    return queue;
}
