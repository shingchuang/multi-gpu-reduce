cmake_minimum_required(VERSION 3.16)
project(multi-gpu-reduce CUDA)

set(CMAKE_CUDA_FLAGS "-gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE on)

add_executable(multi-gpu-reduce multi-gpu-reduce.cu)


set_target_properties(
        multi-gpu-reduce
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)