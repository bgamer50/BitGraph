#pragma once

#include <string>
#include <cuda_runtime.h>

// These are for kernels that heavily access global memory.
// Kernels that make better use of shared memory will have different params
#define BLOCK_SIZE 128
#define NUM_BLOCKS(N) ( (N + BLOCK_SIZE - 1) / BLOCK_SIZE )

inline void cudaCheckErrors(std::string func_name) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error calling %s:\n%s\n\n", func_name.c_str(), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}