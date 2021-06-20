#include <cuda_runtime.h>

/**
    M: (row_ptr, col_ptr) adjacency matrix
    T: intial traversers
    ps: prefix summed result array (indicates index of start for each traverser; last element is total size)
    O: The output array of traversers
    OO: The output origin array (which original traverser did the new traverser originate from)
    N: The original number of traversers
**/
__global__ void k_quadvv_get_adj(int32_t* row_ptr, int32_t* col_ptr, int32_t* T, int32_t* ps, int32_t* O, int32_t* OO, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        int32_t vertex = T[i];
        int output_index = i==0 ? 0 : ps[i - 1];

        int32_t start = row_ptr[vertex];
        int32_t end = row_ptr[vertex + 1];
        
        for(int j = start; j < end; ++j) {
            O[output_index] = col_ptr[j];
            OO[output_index] = vertex;
            ++output_index;
        }
    }
}

/**
    Helper method for gpu_query_adjacency_v_to_v
    For each traverser, calculate the number of new traversers that will be produced.

    M: (row_ptr) adjacency matrix
    T: initial traversers
    R: result array; contains # of new traversers each traverser will generate
    N: # of initial traversers
**/
__global__ void k_quadvv_get_mem(int32_t* row_ptr, int32_t* T, int32_t* R, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        int32_t vertex = T[i];
        int32_t start = row_ptr[vertex];
        int32_t end = row_ptr[vertex + 1];
        
        int32_t v_out_count = end - start;
        R[i] = v_out_count;
    }
}

__global__ void k_prefix_sum(int32_t* A, int32_t* B, int i, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < N; j += stride) {
        if(j < i) B[j] = A[j];
        else B[j] = A[j] + A[j-i];
    }
}