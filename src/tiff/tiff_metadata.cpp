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

    // Populate tile/strip byte offsets for the S3 direct-read path.
    // GDAL exposes these via per-band "MAIN" domain metadata:
    //   BLOCK_OFFSET_tx_ty  – byte offset of the block at tile-column tx, tile-row ty
    //   BLOCK_SIZE_tx_ty    – compressed byte length of that block
    // For PIXEL-interleaved TIFFs band 1 holds all offsets (single IFD).
    // For BAND-planar TIFFs each band has its own IFD entries.
    {
        int num_planes = (info.interleave == "PIXEL") ? 1 : info.samples_per_pixel;
        if (info.is_tiled) {
            int tiles_across = (info.width  + block_width  - 1) / block_width;
            int tiles_down   = (info.height + block_height - 1) / block_height;
            size_t tiles_per_plane = (size_t)tiles_across * tiles_down;
            info.tile_offsets.assign(tiles_per_plane * num_planes, 0);
            info.tile_lengths.assign(tiles_per_plane * num_planes, 0);
            for (int plane = 0; plane < num_planes; ++plane) {
                GDALRasterBand* b = dataset->GetRasterBand(plane + 1);
                for (int ty = 0; ty < tiles_down; ++ty) {
                    for (int tx = 0; tx < tiles_across; ++tx) {
                        const char* off_s = b->GetMetadataItem(
                            CPLSPrintf("BLOCK_OFFSET_%d_%d", tx, ty), "MAIN");
                        const char* len_s = b->GetMetadataItem(
                            CPLSPrintf("BLOCK_SIZE_%d_%d",   tx, ty), "MAIN");
                        size_t idx = (size_t)plane * tiles_per_plane
                                   + (size_t)ty * tiles_across + tx;
                        if (off_s) info.tile_offsets[idx] = std::stoull(off_s);
                        if (len_s) info.tile_lengths[idx]  = std::stoull(len_s);
                    }
                }
            }
        } else {
            // Strip layout: one strip per block_height rows
            int strips_per_plane = (info.height + block_height - 1) / block_height;
            info.strip_offsets.assign((size_t)strips_per_plane * num_planes, 0);
            info.strip_lengths.assign((size_t)strips_per_plane * num_planes, 0);
            for (int plane = 0; plane < num_planes; ++plane) {
                GDALRasterBand* b = dataset->GetRasterBand(plane + 1);
                for (int sy = 0; sy < strips_per_plane; ++sy) {
                    const char* off_s = b->GetMetadataItem(
                        CPLSPrintf("BLOCK_OFFSET_0_%d", sy), "MAIN");
                    const char* len_s = b->GetMetadataItem(
                        CPLSPrintf("BLOCK_SIZE_0_%d",   sy), "MAIN");
                    size_t idx = (size_t)plane * strips_per_plane + sy;
                    if (off_s) info.strip_offsets[idx] = std::stoull(off_s);
                    if (len_s) info.strip_lengths[idx]  = std::stoull(len_s);
                }
            }
        }
    }

    GDALClose(dataset);
    return info;
}



GDALDataset* create_output_dataset(const std::string& output_path, const FileInfo& info,
                                   int nBands) {
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
        nBands, GDT_Float32,
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
