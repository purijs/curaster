#include "texture.h"
#include <stdexcept>

static const char* ALL_FEATURE_NAMES[] = {
    "asm", "contrast", "correlation", "variance", "homogeneity",
    "sum_average", "sum_variance", "sum_entropy", "entropy",
    "diff_variance", "diff_entropy", "dissimilarity", "autocorrelation",
    "max_probability", "cluster_shade", "cluster_prominence", "imc1", "imc2"
};
static constexpr int NUM_ALL_FEATURES = 18;

void build_glcm_params(GLCMParams& gp,
                       const std::vector<std::string>& features,
                       int window,
                       int levels,
                       const std::string& direction_mode,
                       bool log_scale,
                       float val_min,
                       float val_max)
{
    gp.window        = (window % 2 == 0) ? window + 1 : window;
    gp.levels        = levels;
    gp.avg_directions = (direction_mode != "separate");
    gp.log_scale     = log_scale;
    gp.value_min     = val_min;
    gp.value_max     = val_max;
    gp.auto_range    = (val_min == val_max);

    if (features.empty()) {
        gp.features.clear();
        for (int i = 0; i < NUM_ALL_FEATURES; ++i) {
            gp.features.push_back(ALL_FEATURE_NAMES[i]);
        }
    } else {
        gp.features = features;
        for (const auto& f : features) {
            bool found = false;
            for (int i = 0; i < NUM_ALL_FEATURES; ++i) {
                if (f == ALL_FEATURE_NAMES[i]) { found = true; break; }
            }
            if (!found) throw std::runtime_error("Unknown GLCM feature: " + f);
        }
    }

    
    gp.num_output_bands = NUM_ALL_FEATURES;
    if (!gp.avg_directions) {
        gp.num_output_bands = NUM_ALL_FEATURES * 4;
    }
}
