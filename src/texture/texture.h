#pragma once
#include <string>
#include <vector>
#include "../../include/pipeline.h"

void build_glcm_params(GLCMParams& gp,
                       const std::vector<std::string>& features,
                       int window,
                       int levels,
                       const std::string& direction_mode,
                       bool log_scale,
                       float val_min,
                       float val_max);
