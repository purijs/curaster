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
#include "cpl_progress.h"
#include <fstream>
#include <pybind11/pybind11.h>

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
    float* h_master_block = nullptr; 
    std::vector<float*> host_bands_ptrs;
    std::vector<float*> device_bands_ptrs;
    float* h_output_ptr = nullptr;
    float* d_output_ptr = nullptr;
    float** master_device_bands_ptrs = nullptr;
    cudaStream_t stream;

    void alloc(size_t max_bytes_per_band, int num_bands) {
        host_bands_ptrs.resize(num_bands);
        device_bands_ptrs.resize(num_bands);

        cudaCheck(cudaHostAlloc(&h_master_block, max_bytes_per_band * num_bands, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memset(h_master_block, 0, max_bytes_per_band * num_bands);

        for (int i = 0; i < num_bands; i++) {
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

void compute_algebra(const std::string& input_file, const std::string& output_file, const std::string& expression, bool verbose) {
    file_info f;
    f.input_file = input_file.c_str();
    f.output_file = output_file.c_str();
    f.raster_algebric_instructions = program(parse(expression, f.band_indices), f.band_indices);
    
    set_metadata(f); 
    
    size_t available_ram_bytes = get_available_ram();
    size_t total_allowed_ram = (size_t)(available_ram_bytes * 0.60); 
    size_t gdal_cache_bytes = (size_t)(available_ram_bytes * 0.20);  
    size_t pinned_mem_limit = total_allowed_ram - gdal_cache_bytes;  

    if (verbose) {
        std::cout << "[Adaptive Engine] OS Available RAM: " << (available_ram_bytes / (1024*1024*1024.0)) << " GB" << std::endl;
        std::cout << "[Adaptive Engine] Budgeting " << (total_allowed_ram / (1024*1024*1024.0)) << " GB total (60% limit)." << std::endl;
    }

    std::string cache_mb_str = std::to_string(gdal_cache_bytes / (1024 * 1024));
    CPLSetConfigOption("GDAL_CACHEMAX", cache_mb_str.c_str());
    if (verbose) std::cout << "[Adaptive Engine] Set GDAL Cache to: " << cache_mb_str << " MB" << std::endl;

    size_t bytes_per_pixel = (f.band_indices.size() + 1) * sizeof(float); 
    size_t bytes_per_row_total = (size_t)f.width * bytes_per_pixel * num_threads;
    
    int max_safe_rows = pinned_mem_limit / bytes_per_row_total;

    int actual_chunk_height;
    if (f.block_y > 1) {
        actual_chunk_height = f.block_y; 
        while (actual_chunk_height + f.block_y <= max_safe_rows) {
            actual_chunk_height += f.block_y;
        }
    } else {
        actual_chunk_height = std::min(512, max_safe_rows);
    }

    if (actual_chunk_height == 0) {
        actual_chunk_height = f.block_y > 1 ? f.block_y : 1;
        if (verbose) std::cout << "[WARNING] Insufficient RAM to meet 60% rule. Forcing minimum chunk height." << std::endl;
    }

    if (verbose) std::cout << "[Adaptive Engine] Tuned Chunk Height: " << actual_chunk_height << " rows." << std::endl;

    CPLSetConfigOption("GDAL_SWATH_SIZE", "200000000");
    omp_set_num_threads(num_threads);

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    char** flags = nullptr;
    flags = CSLSetNameValue(flags, "COMPRESS", "ZSTD");
    flags = CSLSetNameValue(flags, "ZSTD_LEVEL", "1");
    flags = CSLSetNameValue(flags, "NUM_THREADS", "ALL_CPUS");
    flags = CSLSetNameValue(flags, "TILED", "YES");
    flags = CSLSetNameValue(flags, "BLOCKXSIZE", "512");
    flags = CSLSetNameValue(flags, "BLOCKYSIZE", "512");
    flags = CSLSetNameValue(flags, "INTERLEAVE", "BAND");
    flags = CSLSetNameValue(flags, "BIGTIFF", "IF_SAFER");

    GDALDataset* out_ds = driver->Create(f.output_file, f.width, f.height, 1, GDT_Float32, flags);
    out_ds->SetGeoTransform(f.geoTransform);
    out_ds->SetProjection(f.projection.c_str());
    GDALRasterBand* out_band = out_ds->GetRasterBand(1);
    out_band->SetNoDataValue(-9999.0);
    CSLDestroy(flags);

    GDALDatasetContainer container(num_threads, f.band_indices.size());
    std::vector<int> band_map; 

    for (int i = 0; i < num_threads; ++i) {
        container.in_datasets[i] = (GDALDataset*)GDALOpen(f.input_file, GA_ReadOnly);
        int idx = 0;
        for (int band: f.band_indices) {
            container.bands[i][idx] = container.in_datasets[i]->GetRasterBand(band + 1);
            if (i == 0) band_map.push_back(band + 1); 
            idx++;
        }
    }

    size_t max_pixels = (size_t)f.width * actual_chunk_height;
    size_t max_bytes_per_band = max_pixels * sizeof(float);

    std::vector<ThreadBuffers> pool(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        pool[t].alloc(max_bytes_per_band, f.band_indices.size());
    }

    int num_chunks = (f.height + actual_chunk_height - 1) / actual_chunk_height;

    Instruction* d_program;
    cudaCheck(cudaMalloc(&d_program, f.raster_algebric_instructions.size() * sizeof(Instruction)));
    cudaCheck(cudaMemcpy(d_program, f.raster_algebric_instructions.data(), f.raster_algebric_instructions.size() * sizeof(Instruction), cudaMemcpyHostToDevice));

    std::atomic<int> completed_chunks{0};
    double progress_state = -1.0; 
    
    if (verbose) {
        GDALTermProgress(0.0, nullptr, &progress_state); 
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int tid = omp_get_thread_num();
        int y_offset = chunk * actual_chunk_height;
        int cur_h = std::min(actual_chunk_height, f.height - y_offset);
        size_t pixels = (size_t)f.width * cur_h;

        ThreadBuffers& buf = pool[tid];

        if (chunk + 2 < num_chunks) {
            int adv_y = (chunk + 2) * actual_chunk_height;
            int adv_h = std::min(actual_chunk_height, f.height - adv_y);
            container.in_datasets[tid]->AdviseRead(0, adv_y, f.width, adv_h, f.width, adv_h, GDT_Float32, 0, nullptr, nullptr);
        }
        
        if (f.interleave == "PIXEL") {
            size_t pixel_space = sizeof(float);
            size_t line_space = (size_t)f.width * sizeof(float);
            size_t band_space = max_bytes_per_band;

            CPLErr err = container.in_datasets[tid]->RasterIO(
                GF_Read, 0, y_offset, f.width, cur_h,
                buf.h_master_block,
                f.width, cur_h, GDT_Float32,
                band_map.size(), band_map.data(),
                pixel_space, line_space, band_space, nullptr
            );
        } else {
            for (int b = 0; b < f.band_indices.size(); b++) {
                CPLErr err = container.bands[tid][b]->RasterIO(
                    GF_Read, 0, y_offset, f.width, cur_h, 
                    buf.host_bands_ptrs[b], 
                    f.width, cur_h, GDT_Float32, 0, 0
                );
            }
        }

        size_t groups = (pixels + 3) / 4;
        int blocks = (int)((groups + threadsPerBlock - 1) / threadsPerBlock);

        compute_raster_algebra<<<blocks, threadsPerBlock, 0, buf.stream>>>(
            d_program,
            f.raster_algebric_instructions.size(),
            buf.master_device_bands_ptrs,
            buf.d_output_ptr,
            pixels);

        cudaCheck(cudaStreamSynchronize(buf.stream));

        #pragma omp critical
        {
            (void)out_band->RasterIO(GF_Write, 0, y_offset, f.width, cur_h, buf.h_output_ptr, f.width, cur_h, GDT_Float32, 0, 0);
            
            int current_completed = ++completed_chunks;
            if (verbose) {
                GDALTermProgress(static_cast<double>(current_completed) / num_chunks, nullptr, &progress_state);
                fflush(stdout); 
            }
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        GDALClose(container.in_datasets[i]);
        pool[i].free_all();
    }
    GDALClose(out_ds);
    cudaFree(d_program);
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

    m.def("compute", &compute_algebra, R"pbdoc(
Computes raster algebra from a string expression on the GPU.

This function reads the specified bands from the input raster, processes the 
algebraic expression on the GPU using zero-copy pinned memory, and writes 
the results directly to disk.

Args:
    input_file (str): Path to the input GeoTIFF file.
    output_file (str): Path where the resulting GeoTIFF will be saved.
    expression (str): The algebraic formula (e.g., "(((B1*B1)+(B2*B2))/((B3*B3)+(B4*B4)+1.0))*((B1-B2)/(B3+B4+0.5))+123.45").
    verbose (bool, optional): If True, displays memory telemetry and a progress bar. Defaults to True.
)pbdoc",
        pybind11::arg("input_file"),
        pybind11::arg("output_file"),
        pybind11::arg("expression"),
        pybind11::arg("verbose") = true
    );
}