cmake_minimum_required(VERSION 3.18)

# Set CUDA architecture before project() to avoid detection issues
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(cuTFHE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Add mathdx path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nvidia-mathdx-25.12.0-cuda12/nvidia/mathdx/25.12")
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG)

message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "nanobind found: ${nanobind_VERSION}")

nanobind_add_module(
    _core
    csrc/bindings.cu
)
target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
        -gencode=arch=compute_100,code=sm_100
        --extra-device-vectorization
        --ftz=true
        --ptxas-options=-v
    >
)

target_include_directories(_core PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xla
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nvidia-mathdx-25.12.0-cuda12/nvidia/mathdx/25.12/include
)
target_link_libraries(_core PRIVATE
    CUDA::cudart
    mathdx::cufftdx
)

install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
