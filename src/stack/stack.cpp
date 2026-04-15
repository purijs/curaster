#include "stack.h"
#include "../../include/pipeline.h"
#include "../../include/types.h"

#include "../tiff/tiff_metadata.h"
#include "../chain/chain.h"
#include "../engine/engine.h"
#include "../engine/engine_temporal.h"

#include "gdal_priv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

// ─── StackChain ───────────────────────────────────────────────────────────────

StackChain::StackChain(const std::vector<std::string>& files)
    : files_(files)
{
    if (files.empty()) { throw std::runtime_error("open_stack: empty file list"); }
    file_infos_.resize(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        file_infos_[i] = get_file_info(files[i]);
    }
    verify_alignment();
}

void StackChain::verify_alignment() const {
    const FileInfo& ref = file_infos_[0];
    for (size_t i = 1; i < file_infos_.size(); ++i) {
        const FileInfo& f = file_infos_[i];
        if (f.width != ref.width || f.height != ref.height) {
            throw std::runtime_error(
                "open_stack: scene " + std::to_string(i)
                + " size (" + std::to_string(f.width) + "x" + std::to_string(f.height)
                + ") differs from scene 0 ("
                + std::to_string(ref.width) + "x" + std::to_string(ref.height)
                + "). Reproject first.");
        }
    }
}

std::shared_ptr<StackChain> StackChain::algebra(const std::string& expression) {
    auto s = std::make_shared<StackChain>(*this);
    ChainOp op;
    op.type        = ChainOpType::ALGEBRA;
    op.algebra_expr = expression;
    s->operations_.push_back(op);
    return s;
}

std::shared_ptr<StackChain> StackChain::reproject(const std::string& target_crs,
                                                    double res_x, double res_y,
                                                    const std::string& resampling,
                                                    double nodata)
{
    ReprojectParams rp;
    rp.target_crs   = target_crs;
    rp.pixel_size_x = res_x;
    rp.pixel_size_y = res_y;
    rp.resampling   = (resampling == "nearest") ? ResampleMethod::NEAREST : ResampleMethod::BILINEAR;
    rp.nodata_value = nodata;

    auto s = std::make_shared<StackChain>(*this);
    ChainOp op;
    op.type             = ChainOpType::REPROJECT;
    op.reproject_params = rp;
    s->operations_.push_back(op);
    return s;
}

// ─── StackChain::temporal ────────────────────────────────────────────────────
// Maps the user-facing op string to TemporalOp enum, builds TemporalParams,
// runs engine_temporal, then returns a Chain over an in-memory result.

std::shared_ptr<Chain> StackChain::temporal(const std::string& op_str,
                                              int t0, int t1,
                                              const std::string& /*baseline*/,
                                              const std::vector<float>& time_values)
{
    static const std::unordered_map<std::string, TemporalOp> OP_MAP = {
        {"diff",             TemporalOp::DIFF},
        {"ratio",            TemporalOp::RATIO},
        {"anomaly_mean",     TemporalOp::ANOMALY_MEAN},
        {"anomaly_baseline", TemporalOp::ANOMALY_BASELINE},
        {"trend",            TemporalOp::TREND},
        {"mean",             TemporalOp::TMEAN},
        {"std",              TemporalOp::TSTD},
        {"min",              TemporalOp::TMIN},
        {"max",              TemporalOp::TMAX},
    };

    auto it = OP_MAP.find(op_str);
    if (it == OP_MAP.end()) {
        throw std::runtime_error("StackChain::temporal: unknown op '" + op_str + "'");
    }

    int N = static_cast<int>(files_.size());

    TemporalParams tp;
    tp.op          = it->second;
    tp.t0_idx      = t0;
    tp.t1_idx      = (t1 < 0) ? (N - 1) : t1;
    tp.time_values = time_values;

    // Pre-compute OLS denominator for TREND
    if (tp.op == TemporalOp::TREND) {
        std::vector<float> tv = time_values;
        if (tv.empty()) {
            tv.resize(N);
            std::iota(tv.begin(), tv.end(), 0.f);
        }
        float n = (float)N;
        float sum_t = 0, sum_t2 = 0;
        for (float t : tv) { sum_t += t; sum_t2 += t * t; }
        float denom = n * sum_t2 - sum_t * sum_t;
        tp.denominator  = (std::fabs(denom) > 1e-10f) ? denom : 1.f;
        tp.time_values  = tv; // store the filled version
    }

    // Build a minimal PipelineCtx so engine_temporal can deliver results
    const FileInfo& ref = file_infos_[0];
    size_t total_pixels = (size_t)ref.width * ref.height;

    // Allocate result
    auto result = std::make_shared<RasterResult>();
    result->width      = ref.width;
    result->height     = ref.height;
    result->file_info  = ref;
    result->projection = ref.projection;
    memcpy(result->geo_transform, ref.geo_transform, sizeof(ref.geo_transform));
    result->allocate();

    PipelineCtx ctx;
    ctx.has_temporal         = true;
    ctx.temporal_params      = tp;
    ctx.temporal_num_scenes  = N;
    ctx.temporal_scene_files = files_;
    ctx.result_callback = [&result](int w, int h, float* pixels, int y0) {
        memcpy(result->data.data() + (size_t)y0 * w,
               pixels, (size_t)w * h * sizeof(float));
    };

    run_engine_temporal(files_, ctx, false);

    // Write result to a GDAL /vsimem/ file, return a Chain over it
    // This lets the user chain further ops (.algebra(), .clip(), .save_local()...)
    std::string vsimem_path = "/vsimem/curaster_temporal_"
                             + std::to_string(reinterpret_cast<uintptr_t>(result.get()))
                             + ".tif";

    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* vd = drv->Create(vsimem_path.c_str(),
                                   ref.width, ref.height, 1, GDT_Float32, nullptr);
    vd->SetProjection(ref.projection.c_str());
    vd->SetGeoTransform(const_cast<double*>(ref.geo_transform));
    GDALRasterBand* vb = vd->GetRasterBand(1);
    vb->RasterIO(GF_Write, 0, 0, ref.width, ref.height,
                 result->data.data(), ref.width, ref.height, GDT_Float32, 0, 0);
    GDALClose(vd);

    return std::make_shared<Chain>(vsimem_path)->algebra("B1");
}

std::shared_ptr<RasterResult> StackChain::to_memory(bool verbose) {
    (void)verbose;
    throw std::runtime_error(
        "StackChain::to_memory(): call temporal() first to reduce the stack to a single raster.");
}

void StackChain::save_local(const std::string& /*path*/, bool /*verbose*/) {
    throw std::runtime_error(
        "StackChain::save_local(): call temporal() first to reduce the stack to a single raster.");
}

std::shared_ptr<StackChain> make_stack(const std::vector<std::string>& files) {
    return std::make_shared<StackChain>(files);
}
