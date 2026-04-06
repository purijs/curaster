#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <thread>
#include <omp.h>
#include <cstring>
#include <cuda_runtime.h>
#include <atomic> 
#include "gdal_priv.h"
#include "cpl_string.h"
#include <fstream>
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

const int num_threads = std::thread::hardware_concurrency() - 1;
const int threadsPerBlock = 256;

enum TokenType { NUMBER, OPERATOR, PARENTHESIS, BAND };
enum Opcode { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_LOAD_BAND, OP_LOAD_CONST };

__global__ void compute_raster_algebra(const Instruction* __restrict__ program, int num_instructions, const float* const* __restrict__ bands, float* __restrict__ output, size_t size) {
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t stride = (size_t)gridDim.x * blockDim.x * 4;

    while (base + 3 < size) {
        float4 stack[16]; int sp = 0;
        for (int i = 0; i<num_instructions;i++) {
            Instruction inst = program[i];
            if (inst.op == OP_LOAD_CONST) { stack[sp++] = make_float4(inst.constant, inst.constant, inst.constant, inst.constant); } 
            else if (inst.op == OP_LOAD_BAND) { stack[sp++] = __ldg(reinterpret_cast<const float4*>(bands[inst.band_index] + base)); } 
            else if (inst.op == OP_ADD) { float4 b = stack[--sp]; float4 a = stack[--sp]; stack[sp++] = add_f4(a, b); } 
            else if (inst.op == OP_SUB) { float4 b = stack[--sp]; float4 a = stack[--sp]; stack[sp++] = sub_f4(a, b); } 
            else if (inst.op == OP_MUL) { float4 b = stack[--sp]; float4 a = stack[--sp]; stack[sp++] = mul_f4(a, b); } 
            else if (inst.op == OP_DIV) { float4 b = stack[--sp]; float4 a = stack[--sp]; stack[sp++] = div_f4(a, b); }
        }
        if (sp > 0) { *((float4*)(output + base)) = stack[0]; } else { *((float4*)(output + base)) = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
        base += stride;
    }

    size_t tail_start = (size / 4) * 4;
    size_t tail_tid = tail_start + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tail_tid < size) {
        float stack_tail[16]; int sp = 0;
        for (int i = 0; i < num_instructions; ++i) {
            Instruction inst = program[i];
            if (inst.op == OP_LOAD_CONST) { stack_tail[sp++] = inst.constant; } 
            else if (inst.op == OP_LOAD_BAND) { stack_tail[sp++] = __ldg(&bands[inst.band_index][tail_tid]); } 
            else if (inst.op == OP_ADD) { float right = stack_tail[--sp]; float left = stack_tail[--sp]; stack_tail[sp++] = left + right; } 
            else if (inst.op == OP_SUB) { float right = stack_tail[--sp]; float left = stack_tail[--sp]; stack_tail[sp++] = left - right; } 
            else if (inst.op == OP_MUL) { float right = stack_tail[--sp]; float left = stack_tail[--sp]; stack_tail[sp++] = left * right; } 
            else if (inst.op == OP_DIV) { float right = stack_tail[--sp]; float left = stack_tail[--sp]; stack_tail[sp++] = __fdividef(left, right + 1e-6f); }
        }
        output[tail_tid] = (sp > 0) ? stack_tail[0] : 0.0f;
    }
}

size_t get_available_ram() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t available_kb = 0;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %zu kB", &available_kb);
            break;
        }
    }
    return available_kb * 1024;
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"\nGPU Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Instruction {
    Opcode op;
    float constant;
    int band_index;
};

struct Token {
    TokenType type;
    std::string value;
};

struct ThreadBuffers {
    float* h_master_block = nullptr; // Single contiguous block for GDAL PIXEL scatter-read
    std::vector<float*> host_bands_ptrs;
    std::vector<float*> device_bands_ptrs;
    float* h_output_ptr = nullptr;
    float* d_output_ptr = nullptr;
    float** master_device_bands_ptrs = nullptr;
    cudaStream_t stream;

    void alloc(size_t max_bytes_per_band, int num_bands) {
        host_bands_ptrs.resize(num_bands);
        device_bands_ptrs.resize(num_bands);

        // Allocate ONE massive contiguous block of Pinned RAM
        cudaCheck(cudaHostAlloc(&h_master_block, max_bytes_per_band * num_bands, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(h_master_block, 0, max_bytes_per_band * num_bands);

        for (int i = 0; i < num_bands; i++) {
            // Chop the master block into individual band pointers
            float* h_ptr = h_master_block + (i * (max_bytes_per_band / sizeof(float)));
            float* d_ptr;
            cudaCheck(cudaHostGetDevicePointer((void**)&d_ptr, h_ptr, 0));

            host_bands_ptrs[i] = h_ptr;
            device_bands_ptrs[i] = d_ptr;
        }

        cudaCheck(cudaHostAlloc(&h_output_ptr, max_bytes_per_band, cudaHostAllocMapped));
        cudaCheck(cudaHostGetDevicePointer((void**)&d_output_ptr, h_output_ptr, 0));
        memset(h_output_ptr, 0, max_bytes_per_band);

        cudaCheck(cudaMalloc(&master_device_bands_ptrs, num_bands * sizeof(float*)));
        cudaCheck(cudaMemcpy(master_device_bands_ptrs, device_bands_ptrs.data(), num_bands * sizeof(float*), cudaMemcpyHostToDevice));

        cudaCheck(cudaStreamCreate(&stream));
    }

    void free_all() {
        cudaFreeHost(h_master_block);
        cudaFreeHost(h_output_ptr);
        cudaFree(master_device_bands_ptrs);
        cudaStreamDestroy(stream);
    }
};

struct GDALDatasetContainer {
    std::vector<GDALDataset*> in_datasets;
    std::vector<std::vector<GDALRasterBand*>> bands;

    GDALDatasetContainer(int num_threads, int num_bands) {
        in_datasets.resize(num_threads, nullptr);
        bands.resize(num_threads);
        for (int i = 0; i < num_threads; i++) {
            bands[i].resize(num_bands, nullptr);
        }
    }
};

struct file_info {
    double geoTransform[6];
    int width;
    int height;
    const char* input_file;
    const char* output_file;
    std::string projection;
    std::vector<Instruction> raster_algebric_instructions;
    std::set<int> band_indices;
    
    int block_x;
    int block_y;
    std::string interleave;
    std::string compression;
};

__device__ inline float4 add_f4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ inline float4 sub_f4(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ inline float4 mul_f4(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ inline float4 div_f4(float4 a, float4 b) {
    return make_float4(__fdividef(a.x, b.x + 1e-6f), __fdividef(a.y, b.y + 1e-6f), __fdividef(a.z, b.z + 1e-6f), __fdividef(a.w, b.w + 1e-6f));
}

int get_precedence(const std::string& op) {
    if (op == "+" || op == "-") return 1;
    else if (op == "*" || op == "/") return 2;
    return 0;
}

std::vector<Token> parse(const std::string& expression, std::set<int>& band_indices) {
    std::vector<Token> tokens, operator_stack, output_queue;
    for (int i = 0; i <= expression.length() - 1; i++) {
        if (expression[i] == ' ') [[likely]] continue;
        if (expression[i] == '(' || expression[i] == ')') {
            tokens.push_back({PARENTHESIS, std::string(1, expression[i])});
        } else if (expression[i] == '+' || expression[i] == '-' || expression[i] == '*' || expression[i] == '/') {
            tokens.push_back({OPERATOR, std::string(1, expression[i])});
        } else if (expression[i] == 'B' || expression[i] == 'b') {
            i++; std::string number;
            while (isdigit(expression[i])) { number.push_back(expression[i]); i++; }
            band_indices.insert(std::stoi(number) - 1);
            tokens.push_back({BAND, std::to_string(std::stoi(number) - 1)}); i--;
        } else if (isdigit(expression[i])) {
            std::string number;
            while (isdigit(expression[i])) { number.push_back(expression[i]); i++; }
            tokens.push_back({NUMBER, number}); i--;
        } else [[unlikely]] continue;
    }
    for (int i=0; i<=tokens.size() - 1; i++) {
        if (tokens[i].type == BAND || tokens[i].type == NUMBER) {
            output_queue.push_back(tokens[i]);
        } else if (tokens[i].type == PARENTHESIS) {
            if (tokens[i].value == "(") { operator_stack.push_back(tokens[i]); } 
            else {
                while (operator_stack.size() > 0 && operator_stack.back().value != "(") {
                    output_queue.push_back(operator_stack.back()); operator_stack.pop_back();
                }
                operator_stack.pop_back();
            }
        } else if (tokens[i].type == OPERATOR) {
            while (operator_stack.size() > 0 && operator_stack.back().value != "(" && get_precedence(operator_stack.back().value) >= get_precedence(tokens[i].value)) {
                output_queue.push_back(operator_stack.back()); operator_stack.pop_back();
            }
            operator_stack.push_back(tokens[i]);
        }
    }
    while (operator_stack.size() > 0) { output_queue.push_back(operator_stack.back()); operator_stack.pop_back(); }
    return output_queue;
}

std::vector<Instruction> program(const std::vector<Token>& output_queue, const std::set<int>& band_indices) {
    std::vector<Instruction> instructions;
    for (const auto& token : output_queue) {
        Instruction inst;
        if (token.type == NUMBER) {
            inst.op =  OP_LOAD_CONST; inst.constant = std::stof(token.value); inst.band_index = -1;
        } else if (token.type == BAND) {
            inst.op = OP_LOAD_BAND; inst.constant = 0.0f;
            int global_idx = std::stoi(token.value);
            inst.band_index = std::distance(band_indices.begin(), band_indices.find(global_idx));
        } else if (token.type == OPERATOR) {
            switch (token.value[0]) {
                case '+': inst.op = OP_ADD; break; case '-': inst.op = OP_SUB; break;
                case '*': inst.op = OP_MUL; break; case '/': inst.op = OP_DIV; break;
            }
            inst.constant = 0.0f; inst.band_index = -1;
        }
        instructions.push_back(inst);
    }
    return instructions;
}

void get_input(int argc, char* argv[], file_info& f) {
    if (argc != 4) { std::cerr << "Usage: ./raster <in> <out> <expr>\n"; exit(1); }
    f.input_file = argv[1];
    f.output_file = argv[2];
    f.raster_algebric_instructions = program(parse(std::string(argv[3]), f.band_indices), f.band_indices);
}

void set_metadata(file_info& f) {
    GDALAllRegister();
    GDALDataset* ds = (GDALDataset*)GDALOpen(f.input_file, GA_ReadOnly);
    f.width = ds->GetRasterXSize();
    f.height = ds->GetRasterYSize();
    ds->GetGeoTransform(f.geoTransform);
    const char* proj = ds->GetProjectionRef();
    f.projection = std::string(proj ? proj : "");

    GDALRasterBand* band = ds->GetRasterBand(1);
    band->GetBlockSize(&f.block_x, &f.block_y);

    const char* interleave = ds->GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE");
    f.interleave = interleave ? std::string(interleave) : "BAND";

    const char* compress = ds->GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE");
    f.compression = compress ? std::string(compress) : "UNCOMPRESSED";

    GDALClose(ds);
}

void curaster(std::string& input_file, std::string& output_file, int& red_band_index, int& nir_band_index, int& chunk_size, bool& verbose) {

    const int TILE_HEIGHT = chunk_size;

    GDALAllRegister();
    GDALDataset* rasterfile = (GDALDataset*)GDALOpen(input_file.c_str(), GA_ReadOnly);

    GDALRasterBand* band_red = rasterfile->GetRasterBand(red_band_index);
    GDALRasterBand* band_nir = rasterfile->GetRasterBand(nir_band_index);

    int blockXSize = band_red->GetXSize(), blockYSize = band_red->GetYSize();
    std::vector<float>* buffer_red_pixels = new std::vector<float>(TILE_HEIGHT * blockXSize);
    std::vector<float>* buffer_nir_pixels = new std::vector<float>(TILE_HEIGHT * blockXSize);
    std::vector<float>* buffer_ndvi_pixels = new std::vector<float>(TILE_HEIGHT * blockXSize);

    float* gpu_buffer_red_pixels, *gpu_buffer_nir_pixels, *gpu_buffer_ndvi_pixels;

    cudaMalloc((void**)&gpu_buffer_red_pixels, TILE_HEIGHT * blockXSize * sizeof(float));
    cudaMalloc((void**)&gpu_buffer_nir_pixels, TILE_HEIGHT * blockXSize * sizeof(float));
    cudaMalloc((void**)&gpu_buffer_ndvi_pixels, TILE_HEIGHT * blockXSize * sizeof(float));

    double geoTransform[6];
    rasterfile->GetGeoTransform(geoTransform);
    const char* projection = rasterfile->GetProjectionRef();

    GDALDriver* gtiffDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* outRaster = gtiffDriver->Create(
        output_file.c_str(), blockXSize, blockYSize, 1, GDT_Float32, nullptr
    );
    outRaster->SetGeoTransform(geoTransform);
    outRaster->SetProjection(projection);
    GDALRasterBand* outBand = outRaster->GetRasterBand(1);

    std::cout << "Writing NDVI to disk..." << std::endl;

    double progress = -1.0;
    if (verbose) {
        GDALTermProgress(0.0, nullptr, &progress);
    }

    for (int y_offset = 0; y_offset < blockYSize; y_offset += TILE_HEIGHT) {
        int current_tile_height = std::min(TILE_HEIGHT, blockYSize - y_offset);
        size_t current_pixels = current_tile_height * blockXSize;

        band_red->RasterIO(
            GF_Read,
            0, y_offset,
            blockXSize, current_tile_height,
            buffer_red_pixels->data(),
            blockXSize, current_tile_height,
            GDT_Float32,
            0, 0
        );

        band_nir->RasterIO(
            GF_Read,
            0, y_offset,
            blockXSize, current_tile_height,
            buffer_nir_pixels->data(),
            blockXSize, current_tile_height,
            GDT_Float32,
            0, 0
        );
    
        cudaMemcpy(gpu_buffer_red_pixels, buffer_red_pixels->data(), current_pixels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_buffer_nir_pixels, buffer_nir_pixels->data(), current_pixels * sizeof(float), cudaMemcpyHostToDevice);

        const int blocksPerGrid = (current_pixels + threadsPerBlock - 1) / threadsPerBlock;
        
        compute_ndvi<<<blocksPerGrid, threadsPerBlock>>>(gpu_buffer_red_pixels, gpu_buffer_nir_pixels, gpu_buffer_ndvi_pixels, current_pixels);
        cudaDeviceSynchronize();

        cudaMemcpy(buffer_ndvi_pixels->data(), gpu_buffer_ndvi_pixels, current_pixels * sizeof(float), cudaMemcpyDeviceToHost);

        outRaster->GetRasterBand(1)->RasterIO(
            GF_Write, 0, y_offset, blockXSize, current_tile_height,
            buffer_ndvi_pixels->data(), blockXSize, current_tile_height, GDT_Float32, 0, 0
        );

        if (verbose) {
            double fraction_complete = (double)(y_offset + current_tile_height) / blockYSize;
            GDALTermProgress(fraction_complete, nullptr, &progress);
        }

    }

    cudaFree(gpu_buffer_red_pixels);
    cudaFree(gpu_buffer_nir_pixels);
    cudaFree(gpu_buffer_ndvi_pixels);

    delete buffer_red_pixels;
    delete buffer_nir_pixels;
    delete buffer_ndvi_pixels;

    GDALClose(outRaster);
    GDALClose(rasterfile);
}

PYBIND11_MODULE(curaster, m) {

    m.doc() = "CUDA-accelerated Raster Algebra Calculator.";

    pybind11::module_ os = pybind11::module_::import("os");
    std::string module_dir = os.attr("path").attr("dirname")(m.attr("__file__")).cast<std::string>();
    
    std::string proj_path = module_dir + "/proj_data";
    std::string gdal_path = module_dir + "/gdal_data";
    
    os.attr("environ")["PROJ_LIB"] = proj_path;
    os.attr("environ")["PROJ_DATA"] = proj_path;
    os.attr("environ")["GDAL_DATA"] = gdal_path;

    CPLSetConfigOption("PROJ_LIB", proj_path.c_str());
    CPLSetConfigOption("PROJ_DATA", proj_path.c_str());
    CPLSetConfigOption("GDAL_DATA", gdal_path.c_str());

    m.def("ndvi", &curaster, R"pbdoc(
Computes the Normalized Difference Vegetation Index (NDVI) from a raster file.

This function reads the specified red and near-infrared (NIR) bands from the input
raster, processes the NDVI on the GPU, and writes the results directly to disk.

Args:
    input_file (str): Path to the input GeoTIFF file.
    output_file (str): Path where the resulting NDVI GeoTIFF will be saved.
    red_band_index (int): The index of the Red band.
    nir_band_index (int): The index of the Near-Infrared band.
    chunk_size (int, optional): The height of the chunk to process in memory at once. Defaults to 256.
    verbose (bool, optional): If True, displays a progress bar. Defaults to True.
)pbdoc",
        pybind11::arg("input_file"),
        pybind11::arg("output_file"),
        pybind11::arg("red_band_index"),
        pybind11::arg("nir_band_index"),
        pybind11::arg("chunk_size") = 256,
        pybind11::arg("verbose") = true
    );
}
