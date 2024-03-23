cmake_minimum_required(VERSION 3.27)
project(arrayCUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(arrayCUDA main.cu
        Matrix.cu
        Matrix.cuh
        cudaFunctions.cu
        cudaFunctions.cuh
)

set_target_properties(arrayCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
