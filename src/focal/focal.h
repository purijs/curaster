#pragma once
#include <string>
#include <vector>
#include "../../include/pipeline.h"
#include "../../include/types.h"

void build_focal_params(FocalParams& fp,
                        const std::string& stat,
                        int radius,
                        const std::string& shape,
                        bool clamp_border);

void build_terrain_params(TerrainParams& tp,
                          const std::vector<std::string>& metrics,
                          const std::string& unit,
                          double sun_azimuth,
                          double sun_altitude,
                          const std::string& method,
                          const FileInfo& src_info);
