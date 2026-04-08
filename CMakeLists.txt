cmake_minimum_required(VERSION 3.18...3.28)
project(curaster_project LANGUAGES CXX CUDA)

find_package(Python     COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11   CONFIG REQUIRED)
find_package(GDAL       REQUIRED)
find_package(OpenMP     REQUIRED)
find_package(CURL       REQUIRED)
find_package(OpenSSL    REQUIRED)

# Find ZSTD manually
find_library(ZSTD_LIBRARY  NAMES zstd REQUIRED)
find_path   (ZSTD_INCLUDE  NAMES zstd.h REQUIRED)
if(NOT TARGET ZSTD::ZSTD)
    add_library(ZSTD::ZSTD UNKNOWN IMPORTED)
    set_target_properties(ZSTD::ZSTD PROPERTIES
        IMPORTED_LOCATION             "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE}")
endif()

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES   "75;80;86;89;90")

# ==============================================================================
# THE "STATIC SANDBOX" ARCHITECTURE
# ==============================================================================

# 1. Compile the GPU code completely isolated from Pybind11's macros.
add_library(gpu_math STATIC src/gpu.cu)

# *CRITICAL*: We must compile the static lib with Position Independent Code (PIC)
# so it can be legally linked into the final Python Shared Object (.so) module.
set_target_properties(gpu_math PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(gpu_math PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>: -O3 --use_fast_math -Xptxas=-O3 >
)

# 2. Compile the CPU code into the Pybind11 Python Module.
pybind11_add_module(curaster src/cpu.cpp)

# Tell GCC where to find the CUDA runtime headers (so it can read raster_core.h)
target_include_directories(curaster PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Apply GCC-specific CPU optimizations
target_compile_options(curaster PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp -march=haswell >
)

# 3. Link them together seamlessly
target_link_libraries(curaster PRIVATE
    gpu_math
    GDAL::GDAL
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
    ZSTD::ZSTD
    z
)

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(curaster PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS curaster DESTINATION .)