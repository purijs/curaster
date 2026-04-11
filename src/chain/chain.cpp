/**
 * @file chain.cpp
 * @brief Chain method implementations: pipeline building, context assembly,
 *        and terminal output methods.
 */
#include "chain.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gdal_priv.h"
#include "cpl_vsi.h"

#include "../../include/pipeline.h"
#include "../../include/types.h"
#include "../../include/chunk_queue.h"

#include "../tiff/tiff_metadata.h"
#include "../algebra/algebra_compiler.h"
#include "../clip/clip.h"
#include "../reproject/reproject.h"
#include "../engine/engine.h"

// ─── Construction ─────────────────────────────────────────────────────────────

Chain::Chain(const std::string& input_file)
    : input_file_(input_file) {}

Chain::Chain(const Chain& other)
    : input_file_(other.input_file_)
    , operations_(other.operations_) {}

// ─── Private helpers ─────────────────────────────────────────────────────────

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

// ─── Pipeline context assembly ────────────────────────────────────────────────

/// Build PipelineCtx from the chain's operation list.
static PipelineCtx build_pipeline_context(const std::vector<ChainOp>& operations,
                                           const FileInfo&              src_info,
                                           const std::string&           input_file) {
    PipelineCtx ctx;

    // Identify reprojection (only one is allowed; use the first found).
    for (const auto& op : operations) {
        if (op.type == ChainOpType::REPROJECT) {
            ctx.has_reproject        = true;
            ctx.reproject_params     = op.reproject_params;
            ctx.reproject_output_info = pre_pass_reproject(input_file, src_info,
                                                            op.reproject_params);
            break;
        }
    }

    // For clip operations, scan-convert against the output CRS grid (after reproejct if any).
    const FileInfo& clip_reference_info = ctx.has_reproject
                                        ? ctx.reproject_output_info
                                        : src_info;

    std::vector<int> band_map;
    bool has_algebra = false;

    for (const auto& op : operations) {
        if (op.type == ChainOpType::ALGEBRA) {
            ctx.instructions = compile_algebra_expression(op.algebra_expr, band_map);
            has_algebra = true;
        }
        if (op.type == ChainOpType::CLIP) {
            ctx.has_clip_mask = true;
            parse_polygon_to_spans(op.geojson_str, clip_reference_info, ctx.clip_spans);
        }
    }

    if (!has_algebra) {
        throw std::runtime_error("Chain must contain at least one algebra() operation.");
    }
    if (band_map.empty()) {
        band_map.push_back(0);  // Default: use band 1
    }

    ctx.band_map = band_map;
    return ctx;
}

// ─── execute ──────────────────────────────────────────────────────────────────

void Chain::execute(GDALRasterBand*             output_band,
                    RasterResult*               result,
                    std::shared_ptr<ChunkQueue> chunk_queue,
                    bool                        verbose) {
    FileInfo   src_info = get_file_info(input_file_);
    PipelineCtx ctx     = build_pipeline_context(operations_, src_info, input_file_);
    const FileInfo& out_info = ctx.has_reproject ? ctx.reproject_output_info : src_info;
    (void)out_info; // reserved for future chunk-size calculations

    // Wire up result destinations.
    if (output_band) {
        ctx.output_band = output_band;
    }
    if (result) {
        ctx.result_callback = [result](int width, int height, float* pixels, int y_offset) {
            memcpy(result->data.data() + static_cast<size_t>(y_offset) * width,
                   pixels,
                   static_cast<size_t>(width) * height * sizeof(float));
        };
    }
    if (chunk_queue) {
        ctx.queue_callback = [chunk_queue](int width, int height, float* pixels, int y_offset) {
            ChunkResult chunk;
            chunk.width    = width;
            chunk.height   = height;
            chunk.y_offset = y_offset;
            chunk.data.assign(pixels, pixels + static_cast<size_t>(width) * height);
            chunk_queue->push(std::move(chunk));
        };
    }

    run_engine_ex(input_file_, ctx, verbose);
}

// ─── Builder methods ──────────────────────────────────────────────────────────

std::shared_ptr<Chain> Chain::algebra(const std::string& expression) {
    auto new_chain = std::make_shared<Chain>(*this);
    new_chain->operations_.push_back(
        {ChainOpType::ALGEBRA, expression, "", {}, {}, {}});
    return new_chain;
}

std::shared_ptr<Chain> Chain::clip(const std::string& geojson) {
    auto new_chain = std::make_shared<Chain>(*this);
    new_chain->operations_.push_back(
        {ChainOpType::CLIP, "", geojson, {}, {}, {}});
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
    new_chain->operations_.push_back(
        {ChainOpType::REPROJECT, "", "", {}, {}, rp});
    return new_chain;
}

// ─── Terminal methods ─────────────────────────────────────────────────────────

void Chain::save_local(const std::string& output_path, bool verbose) {
    FileInfo      out_info  = get_output_info();
    GDALDataset*  output_ds = create_output_dataset(output_path, out_info);
    GDALRasterBand* out_band = output_ds->GetRasterBand(1);
    execute(out_band, nullptr, nullptr, verbose);
    GDALClose(output_ds);
}

void Chain::save_s3(const std::string& s3_path, bool verbose) {
    // Materialise to memory first, then upload via GDAL's /vsis3/ virtual FS.
    auto result = to_memory(verbose);

    if (!result || result->data.empty()) {
        throw std::runtime_error("write_s3: empty result — nothing to upload.");
    }

    // Write to a temp file then copy to S3 via CPLCopyFile.
    char tmp_path[] = "/tmp/curaster_XXXXXX.tif";
    int  fd = mkstemps(tmp_path, 4);
    if (fd < 0) {
        throw std::runtime_error("Cannot create temporary file for S3 upload.");
    }
    close(fd);

    GDALDataset* tmp_ds = create_output_dataset(tmp_path, result->file_info);
    (void)tmp_ds->GetRasterBand(1)->RasterIO(
        GF_Write, 0, 0, result->width, result->height,
        result->data.data(), result->width, result->height,
        GDT_Float32, 0, 0);
    GDALClose(tmp_ds);

    if (CPLCopyFile(s3_path.c_str(), tmp_path) != 0) {
        unlink(tmp_path);
        throw std::runtime_error(
            "S3 upload failed. Ensure the path starts with /vsis3/ and AWS credentials are valid.");
    }
    unlink(tmp_path);
}

std::shared_ptr<RasterResult> Chain::to_memory(bool verbose) {
    FileInfo out_info = get_output_info();

    size_t required_bytes   = static_cast<size_t>(out_info.width)
                            * static_cast<size_t>(out_info.height) * sizeof(float);
    size_t safe_ram_limit   = static_cast<size_t>(get_available_ram() * 0.75);

    if (required_bytes > safe_ram_limit) {
        throw std::runtime_error(
            "MemoryError: raster requires " + std::to_string(required_bytes / 1048576)
            + " MB but only " + std::to_string(safe_ram_limit / 1048576)
            + " MB is safely available. Use iter_begin() to stream chunks instead.");
    }

    auto result      = std::make_shared<RasterResult>();
    result->width    = out_info.width;
    result->height   = out_info.height;
    result->file_info   = out_info;
    result->projection  = out_info.projection;
    memcpy(result->geo_transform, out_info.geo_transform, sizeof(out_info.geo_transform));
    result->allocate();

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
            // Swallow exceptions in the background thread; the queue will signal EOF.
        }
        queue->finish();
    }).detach();

    return queue;
}
