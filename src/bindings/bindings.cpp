/**
 * @file bindings.cpp
 * @brief pybind11 Python module definition for curaster.
 *
 * Exposes:
 *  - curaster.open(path)           → Chain
 *  - Chain.algebra(expr)           → Chain
 *  - Chain.clip(geojson)           → Chain
 *  - Chain.reproject(crs, ...)     → Chain
 *  - Chain.get_info()              → dict
 *  - Chain.save_local(path)
 *  - Chain.save_s3(s3_path)
 *  - Chain.to_memory()             → RasterResult
 *  - Chain.iter_begin(buf_chunks)  → ChunkQueue
 *  - RasterResult.width / .height / .proj / .data()  → numpy array
 *  - ChunkQueue.next()             → dict | None
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gdal_priv.h"

#include "../chain/chain.h"
#include "../engine/engine.h"   // init_ram_budget
#include "../../include/types.h"
#include "../../include/chunk_queue.h"

namespace py = pybind11;

PYBIND11_MODULE(curaster, module) {
    GDALAllRegister();
    init_ram_budget();

    // ── curaster.open(path) ───────────────────────────────────────────────
    module.def("open", [](const std::string& file_path) {
        return std::make_shared<Chain>(file_path);
    }, py::arg("path"),
       "Open a GeoTIFF (local path or S3 URI) and return a lazy Chain.");

    // ── Chain ─────────────────────────────────────────────────────────────
    py::class_<Chain, std::shared_ptr<Chain>>(module, "Chain")
        .def("algebra",   &Chain::algebra,  py::arg("expression"),
             "Append a band-math algebra operation (e.g. '(B1 - B2) / B3').")
        .def("clip",      &Chain::clip,     py::arg("geojson"),
             "Append a GeoJSON polygon clip.")
        .def("get_info",  [](const Chain& chain) {
             // Build the Python dict here so Chain itself has no pybind11 dependency.
             FileInfo out_info = chain.get_output_info();
             py::list geotransform;
             for (int i = 0; i < 6; ++i) { geotransform.append(out_info.geo_transform[i]); }
             py::dict info;
             info["width"]        = out_info.width;
             info["height"]       = out_info.height;
             info["geotransform"] = geotransform;
             info["crs"]          = out_info.projection;
             return info;
         }, "Return dict {width, height, geotransform, crs} of the output raster.")
        .def("reproject", &Chain::reproject,
             py::arg("target_crs"),
             py::arg("res_x")       = 0.0,
             py::arg("res_y")       = 0.0,
             py::arg("resampling")  = "bilinear",
             py::arg("nodata")      = -9999.0,
             py::arg("te_xmin")     = 0.0,
             py::arg("te_ymin")     = 0.0,
             py::arg("te_xmax")     = 0.0,
             py::arg("te_ymax")     = 0.0,
             "Append a reprojection to target_crs.")
        .def("save_local", &Chain::save_local,
             py::arg("path"), py::arg("verbose") = false,
             "Execute and write output to a local GeoTIFF.")
        .def("save_s3",    &Chain::save_s3,
             py::arg("s3_path"), py::arg("verbose") = false,
             "Execute and upload output to an S3 /vsis3/ path.")
        .def("to_memory",  &Chain::to_memory,
             py::arg("verbose") = false,
             "Execute and return a RasterResult holding all pixels.")
        .def("iter_begin", &Chain::iter_begin,
             py::arg("buf_chunks") = 4,
             "Execute in background; returns a ChunkQueue for streaming.");

    // ── RasterResult ──────────────────────────────────────────────────────
    py::class_<RasterResult, std::shared_ptr<RasterResult>>(module, "RasterResult")
        .def_readonly("width",  &RasterResult::width,      "Output width in pixels.")
        .def_readonly("height", &RasterResult::height,     "Output height in pixels.")
        .def_readonly("proj",   &RasterResult::projection, "WKT coordinate reference system.")
        .def("data", [](const RasterResult& result) {
            // Return a 2-D (height × width) numpy array backed by the result's memory.
            return py::array_t<float>(
                { static_cast<py::ssize_t>(result.height),
                  static_cast<py::ssize_t>(result.width) },
                result.data.data());
        }, "Return pixel data as a (height, width) float32 numpy array.");

    // ── ChunkQueue ────────────────────────────────────────────────────────
    py::class_<ChunkQueue, std::shared_ptr<ChunkQueue>>(module, "ChunkQueue")
        .def("next", [](ChunkQueue& queue) -> py::object {
            ChunkResult chunk;
            if (!queue.pop(chunk)) {
                return py::none();  // EOF: no more chunks
            }

            // Transfer ownership of the pixel buffer to a capsule so numpy
            // does not need to copy the data.
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
