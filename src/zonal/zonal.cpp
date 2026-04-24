#include "zonal.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <ogr_geometry.h>
#include <ogr_spatialref.h>
#include "gdal_priv.h"






static void scan_convert_ring(
    const std::vector<double>& px,
    const std::vector<double>& py,
    const FileInfo&            file_info,
    int                        chunk_y0,
    int                        chunk_height,
    uint16_t                   zone_id,
    uint16_t*                  out_labels)
{
    int npts = static_cast<int>(px.size());
    if (npts < 2) return;

    double py_min = *std::min_element(py.begin(), py.end());
    double py_max = *std::max_element(py.begin(), py.end());

    int row_min = std::max(chunk_y0, (int)py_min);
    int row_max = std::min(chunk_y0 + chunk_height - 1, (int)py_max);
    if (row_min > row_max) return;

    for (int row = row_min; row <= row_max; ++row) {
        std::vector<double> xs;
        for (int e = 0; e < npts - 1; ++e) {
            double ey0 = py[e], ey1 = py[e+1];
            double ex0 = px[e], ex1 = px[e+1];
            bool crosses = (ey0 <= row && row < ey1) || (ey1 <= row && row < ey0);
            if (crosses) {
                double x_at = ex0 + (row - ey0) / (ey1 - ey0) * (ex1 - ex0);
                xs.push_back(x_at);
            }
        }
        std::sort(xs.begin(), xs.end());
        for (size_t i = 0; i + 1 < xs.size(); i += 2) {
            int c0 = std::max(0,                    (int)xs[i]);
            int c1 = std::min(file_info.width - 1,  (int)xs[i+1]);
            int local_row = row - chunk_y0;
            for (int c = c0; c <= c1; ++c) {
                out_labels[local_row * file_info.width + c] = zone_id;
            }
        }
    }
}








std::vector<PrebuiltZone> build_prebuilt_zones(
    const std::string& geojson_str,
    const FileInfo&    file_info)
{
    OGRGeometry* geom = OGRGeometryFactory::createFromGeoJson(geojson_str.c_str());
    if (!geom) throw std::runtime_error("zonal_stats: invalid GeoJSON");

    double inv_gt[6];
    GDALInvGeoTransform(const_cast<double*>(file_info.geo_transform), inv_gt);

    std::vector<PrebuiltZone> zones;

    auto add_polygon = [&](OGRPolygon* poly, uint16_t id) {
        OGRLinearRing* ring = poly->getExteriorRing();
        int npts = ring->getNumPoints();
        PrebuiltZone z;
        z.id = id;
        z.px.resize(npts);
        z.py.resize(npts);
        for (int i = 0; i < npts; ++i) {
            double gx = ring->getX(i), gy = ring->getY(i);
            z.px[i] = inv_gt[0] + gx * inv_gt[1] + gy * inv_gt[2];
            z.py[i] = inv_gt[3] + gx * inv_gt[4] + gy * inv_gt[5];
        }
        zones.push_back(std::move(z));
    };

    OGRwkbGeometryType gt = wkbFlatten(geom->getGeometryType());
    if (gt == wkbPolygon) {
        add_polygon(static_cast<OGRPolygon*>(geom), 1);
    } else if (gt == wkbMultiPolygon) {
        auto* mp = static_cast<OGRMultiPolygon*>(geom);
        for (int i = 0; i < mp->getNumGeometries(); ++i) {
            add_polygon(static_cast<OGRPolygon*>(mp->getGeometryRef(i)),
                        static_cast<uint16_t>(i + 1));
        }
    } else if (gt == wkbGeometryCollection) {
        auto* gc = static_cast<OGRGeometryCollection*>(geom);
        int   idx = 0;
        for (int i = 0; i < gc->getNumGeometries(); ++i) {
            OGRGeometry* g = gc->getGeometryRef(i);
            if (wkbFlatten(g->getGeometryType()) == wkbPolygon) {
                add_polygon(static_cast<OGRPolygon*>(g),
                            static_cast<uint16_t>(++idx));
            }
        }
    }

    OGRGeometryFactory::destroyGeometry(geom);
    return zones;
}













void rasterize_zones_prebuilt(
    const std::vector<PrebuiltZone>& zones,
    const FileInfo&                  file_info,
    int                              chunk_y0,
    int                              chunk_height,
    uint16_t*                        out_labels)
{
    size_t label_count = static_cast<size_t>(file_info.width) * chunk_height;
    std::fill(out_labels, out_labels + label_count, (uint16_t)0);

    for (const auto& z : zones) {
        scan_convert_ring(z.px, z.py, file_info, chunk_y0, chunk_height,
                          z.id, out_labels);
    }
}



int rasterize_zones_chunked(
    const std::string& geojson_str,
    const FileInfo&    file_info,
    int                chunk_y0,
    int                chunk_height,
    uint16_t*          out_labels)
{
    auto zones = build_prebuilt_zones(geojson_str, file_info);
    rasterize_zones_prebuilt(zones, file_info, chunk_y0, chunk_height, out_labels);
    return static_cast<int>(zones.size());
}

int count_zones_geojson(const std::string& geojson_str) {
    OGRGeometry* geom = OGRGeometryFactory::createFromGeoJson(geojson_str.c_str());
    if (!geom) return 0;
    int n = 0;
    OGRwkbGeometryType gt = wkbFlatten(geom->getGeometryType());
    if (gt == wkbPolygon) {
        n = 1;
    } else if (gt == wkbMultiPolygon) {
        n = static_cast<OGRMultiPolygon*>(geom)->getNumGeometries();
    } else if (gt == wkbGeometryCollection) {
        auto* gc = static_cast<OGRGeometryCollection*>(geom);
        for (int i = 0; i < gc->getNumGeometries(); ++i) {
            if (wkbFlatten(gc->getGeometryRef(i)->getGeometryType()) == wkbPolygon) ++n;
        }
    }
    OGRGeometryFactory::destroyGeometry(geom);
    return n;
}



std::vector<ZoneResult> aggregate_zonal_results(
    const int*   h_count,
    const float* h_sum,
    const float* h_sum_sq,
    const float* h_min,
    const float* h_max,
    int          num_zones,
    const std::vector<std::string>& )
{
    std::vector<ZoneResult> results;
    results.reserve(num_zones);

    for (int z = 1; z <= num_zones; ++z) {
        ZoneResult r;
        r.zone_id = z;
        r.count   = h_count[z];
        r.sum     = h_sum[z];
        r.min_val = h_min[z];
        r.max_val = h_max[z];

        if (r.count > 0) {
            r.mean = r.sum / (double)r.count;
            double var = (double)h_sum_sq[z] / r.count - r.mean * r.mean;
            r.std_dev = std::sqrt(std::max(0.0, var));
        }
        results.push_back(r);
    }
    return results;
}
