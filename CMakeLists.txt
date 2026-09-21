cmake_minimum_required(VERSION 3.18)
project(gpuxray_nccl_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS
        $ENV{NCCL_HOME}/include
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
)

find_library(NCCL_LIBRARY nccl
    HINTS
        $ENV{NCCL_HOME}/lib
        $ENV{NCCL_HOME}/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/cuda/lib64
)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR
        "NCCL not found. Set NCCL_HOME or provide NCCL include/library paths.")
endif()

add_executable(gpuxray_nccl_bench
    src/main.cu
)

target_include_directories(gpuxray_nccl_bench PRIVATE
    ${NCCL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gpuxray_nccl_bench PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARY}
)

set_target_properties(gpuxray_nccl_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
