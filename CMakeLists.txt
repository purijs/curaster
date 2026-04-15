cmake_minimum_required(VERSION 3.18...3.28)
project(curaster_project LANGUAGES CXX CUDA)

find_package(Python   COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(GDAL    REQUIRED)
find_package(OpenMP  COMPONENTS CXX REQUIRED)
find_package(CURL    REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB    REQUIRED)

find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
find_path(ZSTD_INCLUDE    NAMES zstd.h REQUIRED)

if(NOT TARGET ZSTD::ZSTD)
    add_library(ZSTD::ZSTD UNKNOWN IMPORTED)
    set_target_properties(ZSTD::ZSTD PROPERTIES
        IMPORTED_LOCATION             "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE}")
endif()

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

add_library(gpu_kernels STATIC
    src/gpu/gpu.cu
    src/gpu/gpu_warp.cu
    src/gpu/gpu_focal.cu
    src/gpu/gpu_glcm.cu
    src/gpu/gpu_zonal.cu)

set_target_properties(gpu_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

target_include_directories(gpu_kernels PRIVATE
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_options(gpu_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xptxas=-O3
    >)

pybind11_add_module(curaster
    src/s3/s3_auth.cpp
    src/s3/s3_fetch.cpp

    src/tiff/tiff_metadata.cpp
    src/tiff/decompression.cpp

    src/tile_io/tile_io.cpp

    src/algebra/algebra_compiler.cpp

    src/clip/clip.cpp

    src/focal/focal.cpp

    src/texture/texture.cpp

    src/zonal/zonal.cpp

    src/stack/stack.cpp

    src/reproject/reproject.cpp

    src/engine/engine.cpp
    src/engine/engine_reproject.cpp
    src/engine/engine_focal.cpp
    src/engine/engine_zonal.cpp
    src/engine/engine_temporal.cpp

    src/chain/chain.cpp

    src/bindings/bindings.cpp)

target_include_directories(curaster PRIVATE
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${GDAL_INCLUDE_DIRS})

target_compile_options(curaster PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:
        /O2
        /arch:AVX2
    >>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        -O3
        -march=haswell
    >>)

target_link_libraries(curaster PRIVATE
    gpu_kernels
    GDAL::GDAL
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
    ZSTD::ZSTD
    ZLIB::ZLIB)

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(curaster PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS curaster DESTINATION .)