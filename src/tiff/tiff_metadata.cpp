/**
 * @file tiff_metadata.cpp
 * @brief GeoTIFF metadata extraction and output dataset creation.
 */
#include "tiff_metadata.h"

#include <stdexcept>
#include <string>

#include "gdal_priv.h"
#include "cpl_string.h"



FileInfo get_file_info(const std::string& file_path) {
    GDALAllRegister();

    GDALDataset* dataset = static_cast<GDALDataset*>(
        GDALOpen(file_path.c_str(), GA_ReadOnly));
    if (!dataset) {
        throw std::runtime_error("Cannot open: " + file_path);
    }

    FileInfo info;

    
    info.width  = dataset->GetRasterXSize();
    info.height = dataset->GetRasterYSize();
    info.samples_per_pixel = dataset->GetRasterCount();

    
    double gt[6];
    dataset->GetGeoTransform(gt);
    memcpy(info.geo_transform, gt, sizeof(gt));

    const char* wkt = dataset->GetProjectionRef();
    info.projection = wkt ? wkt : "";

    
    GDALRasterBand* first_band = dataset->GetRasterBand(1);
    info.data_type = static_cast<int>(first_band->GetRasterDataType());

    int block_width  = 512;
    int block_height = 512;
    first_band->GetBlockSize(&block_width, &block_height);
    info.tile_width  = block_width;
    info.tile_height = block_height;
    info.is_tiled    = (block_width != info.width);

    if (!info.is_tiled) {
        info.rows_per_strip = block_height;
    }

    
    const char* interleave_str = dataset->GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE");
    info.interleave = interleave_str ? interleave_str : "BAND";

    const char* compression_str = dataset->GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE");
    info.compression = compression_str ? compression_str : "NONE";

    
    char** structure_meta = dataset->GetMetadata("IMAGE_STRUCTURE");
    if (structure_meta) {
        for (int i = 0; structure_meta[i] != nullptr; ++i) {
            std::string entry = structure_meta[i];
            if (entry.find("PREDICTOR=") == 0) {
                info.predictor = std::stoi(entry.substr(10));
            }
        }
    }

    GDALClose(dataset);
    return info;
}



GDALDataset* create_output_dataset(const std::string& output_path, const FileInfo& info) {
    GDALAllRegister();

    auto* driver = static_cast<GDALDriver*>(GDALGetDriverByName("GTiff"));

    
    char** creation_options = nullptr;
    creation_options = CSLSetNameValue(creation_options, "TILED",     "YES");
    creation_options = CSLSetNameValue(creation_options, "BLOCKXSIZE","512");
    creation_options = CSLSetNameValue(creation_options, "BLOCKYSIZE","512");
    creation_options = CSLSetNameValue(creation_options, "COMPRESS",  "DEFLATE");
    creation_options = CSLSetNameValue(creation_options, "PREDICTOR", "3");
    creation_options = CSLSetNameValue(creation_options, "BIGTIFF",   "YES");

    GDALDataset* output_dataset = driver->Create(
        output_path.c_str(),
        info.width, info.height,
        1, GDT_Float32,
        creation_options);

    CSLDestroy(creation_options);

    if (!output_dataset) {
        throw std::runtime_error("Cannot create output: " + output_path);
    }

    output_dataset->SetGeoTransform(const_cast<double*>(info.geo_transform));
    if (!info.projection.empty()) {
        output_dataset->SetProjection(info.projection.c_str());
    }

    return output_dataset;
}
