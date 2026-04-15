#pragma once
#include <string>
#include "../../include/pipeline.h"

void run_engine_temporal(const std::vector<std::string>& scene_files,
                          PipelineCtx& ctx,
                          bool verbose);
