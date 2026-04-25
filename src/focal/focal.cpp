#include "focal.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <ogr_spatialref.h>

void build_focal_params(FocalParams& fp,
                        const std::string& stat,
                        int radius,
                        const std::string& shape,
                        bool clamp_border)
{
    fp.radius       = radius;
    fp.clamp_border = clamp_border;

    if      (stat == "mean")     fp.stat = FocalStat::MEAN;
    else if (stat == "sum")      fp.stat = FocalStat::SUM;
    else if (stat == "min")      fp.stat = FocalStat::MIN;
    else if (stat == "max")      fp.stat = FocalStat::MAX;
    else if (stat == "std")      fp.stat = FocalStat::STD;
    else if (stat == "variance") fp.stat = FocalStat::VARIANCE;
    else if (stat == "median")   fp.stat = FocalStat::MEDIAN;
    else if (stat == "range")    fp.stat = FocalStat::RANGE;
    else throw std::runtime_error("Unknown focal stat: " + stat);

    if      (shape == "square") fp.shape = FocalShape::SQUARE;
    else if (shape == "circle") fp.shape = FocalShape::CIRCLE;
    else throw std::runtime_error("Unknown focal shape: " + shape);
}

void build_terrain_params(TerrainParams& tp,
                          const std::vector<std::string>& metrics,
                          const std::string& unit,
                          double sun_azimuth,
                          double sun_altitude,
                          const std::string& method,
                          const FileInfo& src_info)
{
    tp.metrics      = metrics;
    tp.unit         = unit;
    tp.sun_azimuth  = sun_azimuth;
    tp.sun_altitude = sun_altitude;
    tp.method       = method;
    tp.features_mask = 0;
    tp.num_output_bands = 0;

    
    double gt1 = src_info.geo_transform[1];
    double gt2 = src_info.geo_transform[2];
    double gt4 = src_info.geo_transform[4];
    double gt5 = src_info.geo_transform[5];

    double cs_x = std::sqrt(gt1*gt1 + gt4*gt4);
    double cs_y = std::sqrt(gt2*gt2 + gt5*gt5);

    
    if (!src_info.projection.empty()) {
        OGRSpatialReference srs;
        srs.SetFromUserInput(src_info.projection.c_str());
        if (srs.IsGeographic()) {
            double lat_center = src_info.geo_transform[3]
                              + (src_info.height / 2.0) * gt5;
            double lat_rad = lat_center * M_PI / 180.0;
            cs_x = cs_x * std::cos(lat_rad) * 111320.0;
            cs_y = cs_y * 111320.0;
        }
    }

    tp.cell_size_x = static_cast<float>(cs_x);
    tp.cell_size_y = static_cast<float>(cs_y);

    for (const auto& m : metrics) {
        if      (m == "slope")        { tp.features_mask |= TERRAIN_SLOPE;     ++tp.num_output_bands; }
        else if (m == "aspect")       { tp.features_mask |= TERRAIN_ASPECT;    ++tp.num_output_bands; }
        else if (m == "hillshade")    { tp.features_mask |= TERRAIN_HILLSHADE; ++tp.num_output_bands; }
        else if (m == "tri")          { tp.features_mask |= TERRAIN_TRI;       ++tp.num_output_bands; }
        else if (m == "tpi")          { tp.features_mask |= TERRAIN_TPI;       ++tp.num_output_bands; }
        else if (m == "roughness")    { tp.features_mask |= TERRAIN_ROUGHNESS; ++tp.num_output_bands; }
        else if (m == "prof_curv")    { tp.features_mask |= TERRAIN_PROF_CURV;  ++tp.num_output_bands; }
        else if (m == "plan_curv")    { tp.features_mask |= TERRAIN_PLAN_CURV;  ++tp.num_output_bands; }
        else if (m == "total_curv")   { tp.features_mask |= TERRAIN_TOTAL_CURV; ++tp.num_output_bands; }
        else throw std::runtime_error("Unknown terrain metric: " + m);
    }
    if (tp.num_output_bands == 0) tp.num_output_bands = 1;
}
