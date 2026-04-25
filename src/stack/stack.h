#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../../include/pipeline.h"  
#include "../../include/types.h"     

class Chain;

class StackChain {
public:
    explicit StackChain(const std::vector<std::string>& files);

    std::shared_ptr<StackChain> algebra(const std::string& expression);
    std::shared_ptr<StackChain> reproject(const std::string& target_crs,
                                           double res_x = 0.0, double res_y = 0.0,
                                           const std::string& resampling = "bilinear",
                                           double nodata = -9999.0);

    
    
    std::shared_ptr<Chain> temporal(const std::string& op,
                                     int t0 = 0, int t1 = -1,
                                     const std::string& baseline = "mean",
                                     const std::vector<float>& time_values = {});

    std::shared_ptr<RasterResult> to_memory(bool verbose = false);
    void save_local(const std::string& path, bool verbose = false);

private:
    std::vector<std::string>  files_;
    std::vector<FileInfo>     file_infos_;
    std::vector<ChainOp>      operations_;

    void verify_alignment() const;
};

std::shared_ptr<StackChain> make_stack(const std::vector<std::string>& files);
