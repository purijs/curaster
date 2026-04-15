#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gdal_priv.h"

#include "../chain/chain.h"
#include "../stack/stack.h"
#include "../engine/engine.h"
#include "../../include/types.h"
#include "../../include/chunk_queue.h"

namespace py = pybind11;

PYBIND11_MODULE(curaster, module) {
    GDALAllRegister();
    init_ram_budget();

    module.def("open", [](const std::string& file_path) {
        return std::make_shared<Chain>(file_path);
    }, py::arg("path"), "Open a GeoTIFF and return a lazy Chain.");

    module.def("open_stack", [](const std::vector<std::string>& files) {
        return make_stack(files);
    }, py::arg("files"), "Open multiple co-registered GeoTIFFs as a StackChain.");

    // ── Chain ─────────────────────────────────────────────────────────────────
    py::class_<Chain, std::shared_ptr<Chain>>(module, "Chain")
        .def("algebra",   &Chain::algebra,  py::arg("expression"))
        .def("clip",      &Chain::clip,     py::arg("geojson"))
        .def("get_info",  [](const Chain& chain) {
             FileInfo out_info = chain.get_output_info();
             py::list geotransform;
             for (int i = 0; i < 6; ++i) { geotransform.append(out_info.geo_transform[i]); }
             py::dict info;
             info["width"]        = out_info.width;
             info["height"]       = out_info.height;
             info["geotransform"] = geotransform;
             info["crs"]          = out_info.projection;
             return info;
         })
        .def("reproject", &Chain::reproject,
             py::arg("target_crs"),
             py::arg("res_x")      = 0.0,
             py::arg("res_y")      = 0.0,
             py::arg("resampling") = "bilinear",
             py::arg("nodata")     = -9999.0,
             py::arg("te_xmin")    = 0.0,
             py::arg("te_ymin")    = 0.0,
             py::arg("te_xmax")    = 0.0,
             py::arg("te_ymax")    = 0.0)
        .def("focal",   &Chain::focal,
             py::arg("stat")         = "mean",
             py::arg("radius")       = 1,
             py::arg("shape")        = "square",
             py::arg("clamp_border") = true,
             "Focal (neighborhood) statistics. stat: mean/sum/min/max/std/variance/median/range.")
        .def("terrain", &Chain::terrain,
             py::arg("metrics")      = std::vector<std::string>{},
             py::arg("unit")         = "degrees",
             py::arg("sun_azimuth")  = 315.0,
             py::arg("sun_altitude") = 45.0,
             py::arg("method")       = "horn",
             "Terrain analysis. metrics: slope/aspect/hillshade/tri/tpi/roughness/prof_curv/plan_curv.")
        .def("texture", &Chain::texture,
             py::arg("features")        = std::vector<std::string>{},
             py::arg("window")          = 11,
             py::arg("levels")          = 32,
             py::arg("direction_mode")  = "average",
             py::arg("log_scale")       = false,
             py::arg("val_min")         = 0.f,
             py::arg("val_max")         = 0.f,
             "GLCM texture features. Returns 18-band raster with Haralick features.")
        .def("zonal_stats", &Chain::zonal_stats,
             py::arg("geojson"),
             py::arg("stats")   = std::vector<std::string>{},
             py::arg("band")    = 1,
             py::arg("verbose") = false,
             "Compute zonal statistics. Returns list of dicts with zone stats.")
        .def("save_local", &Chain::save_local,
             py::arg("path"), py::arg("verbose") = false)
        .def("save_s3",    &Chain::save_s3,
             py::arg("s3_path"), py::arg("verbose") = false)
        .def("to_memory",  &Chain::to_memory,
             py::arg("verbose") = false)
        .def("iter_begin", &Chain::iter_begin,
             py::arg("buf_chunks") = 4);

    // ── StackChain ────────────────────────────────────────────────────────────
    py::class_<StackChain, std::shared_ptr<StackChain>>(module, "StackChain")
        .def("algebra",  &StackChain::algebra,  py::arg("expression"))
        .def("reproject", &StackChain::reproject,
             py::arg("target_crs"),
             py::arg("res_x")      = 0.0,
             py::arg("res_y")      = 0.0,
             py::arg("resampling") = "bilinear",
             py::arg("nodata")     = -9999.0)
        .def("temporal", &StackChain::temporal,
             py::arg("op"),
             py::arg("t0")          = 0,
             py::arg("t1")          = -1,
             py::arg("baseline")    = "mean",
             py::arg("time_values") = std::vector<float>{},
             "Reduce the scene stack to a single output via a temporal operation.\n"
             "op: diff, ratio, anomaly_mean, trend, mean, std, min, max");

    // ── RasterResult ──────────────────────────────────────────────────────────
    py::class_<RasterResult, std::shared_ptr<RasterResult>>(module, "RasterResult")
        .def_readonly("width",  &RasterResult::width)
        .def_readonly("height", &RasterResult::height)
        .def_readonly("proj",   &RasterResult::projection)
        .def("data", [](const RasterResult& result) {
            return py::array_t<float>(
                { static_cast<py::ssize_t>(result.height),
                  static_cast<py::ssize_t>(result.width) },
                result.data.data());
        });

    // ── ZoneResult ────────────────────────────────────────────────────────────
    py::class_<ZoneResult>(module, "ZoneResult")
        .def_readonly("zone_id", &ZoneResult::zone_id)
        .def_readonly("mean",    &ZoneResult::mean)
        .def_readonly("min",     &ZoneResult::min_val)
        .def_readonly("max",     &ZoneResult::max_val)
        .def_readonly("std_dev", &ZoneResult::std_dev)
        .def_readonly("count",   &ZoneResult::count)
        .def_readonly("sum",     &ZoneResult::sum)
        .def("to_dict", [](const ZoneResult& r) {
            py::dict d;
            d["zone_id"] = r.zone_id;
            d["mean"]    = r.mean;
            d["min"]     = r.min_val;
            d["max"]     = r.max_val;
            d["std_dev"] = r.std_dev;
            d["count"]   = r.count;
            d["sum"]     = r.sum;
            return d;
        });

    // ── ChunkQueue ────────────────────────────────────────────────────────────
    py::class_<ChunkQueue, std::shared_ptr<ChunkQueue>>(module, "ChunkQueue")
        .def("next", [](ChunkQueue& queue) -> py::object {
            ChunkResult chunk;
            if (!queue.pop(chunk)) { return py::none(); }

            auto* owned_buffer = new std::vector<float>(std::move(chunk.data));
            py::capsule buffer_capsule(owned_buffer, [](void* ptr) noexcept {
                delete static_cast<std::vector<float>*>(ptr);
            });

            py::array_t<float> array(
                { static_cast<py::ssize_t>(chunk.height),
                  static_cast<py::ssize_t>(chunk.width) },
                owned_buffer->data(),
                buffer_capsule);

            py::dict result_dict;
            result_dict["y_offset"] = chunk.y_offset;
            result_dict["width"]    = chunk.width;
            result_dict["height"]   = chunk.height;
            result_dict["data"]     = array;
            return result_dict;
        }, "Return the next chunk dict {y_offset, width, height, data} or None at EOF.");
}
