cmake_minimum_required(VERSION 3.18...3.28)
project(curaster_project LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(GDAL REQUIRED)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

pybind11_add_module(curaster MODULE src/raster.cu)

target_link_libraries(curaster PRIVATE GDAL::GDAL)

install(TARGETS curaster DESTINATION .)

if(UNIX)
    # Ubuntu Docker paths
    install(DIRECTORY /usr/share/proj/ DESTINATION proj_data)
    install(DIRECTORY /usr/share/gdal/ DESTINATION gdal_data)
elseif(WIN32)
    # Windows Conda paths
    install(DIRECTORY $ENV{CONDA_PREFIX}/Library/share/proj/ DESTINATION proj_data)
    install(DIRECTORY $ENV{CONDA_PREFIX}/Library/share/gdal/ DESTINATION gdal_data)
endif()
