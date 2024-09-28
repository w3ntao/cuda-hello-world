cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_hello_world CUDA)

set(CMAKE_CUDA_STANDARD 17)

if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    # required by CLion
endif ()
message("CUDA compiler: ${CMAKE_CUDA_COMPILER}")

add_executable(cuda_hello_world main.cu)

set_target_properties(
        cuda_hello_world PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
)
