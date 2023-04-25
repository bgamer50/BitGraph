#include "algorithm/ConnectedComponentsGPUGraphAlgorithm.cuh"
#include "structure/GPUGraph.cuh"
#include "step/gpu/GPUTraversalHelper.cuh"
#include <cuda_runtime.h>

__global__ void k_cc(size_t N, size_t* row_ptr, size_t* col_ptr, size_t* old_cc, size_t* new_cc);
__global__ void k_sub(size_t N, size_t* A, size_t* B);
__global__ void k_cc_init(size_t N, size_t* cc);

std::unordered_map<std::string, boost::any> ConnectedComponentsGPUGraphAlgorithm::exec(GPUGraph* graph) {
    bitgraph::matrix::sparse_matrix_device adjacency_matrix = graph->get_adjacency_matrix(this->direction);

    size_t* new_cc = nullptr; cudaMalloc((void**) &new_cc, sizeof(size_t) * adjacency_matrix.num_rows);
    cudaDeviceSynchronize();
    cudaCheckErrors("malloc new cc");
    size_t* old_cc = nullptr; cudaMalloc((void**) &old_cc, sizeof(size_t) * adjacency_matrix.num_rows);
    cudaDeviceSynchronize();
    cudaCheckErrors("malloc old cc");

    k_cc_init<<<NUM_BLOCKS(adjacency_matrix.num_rows), BLOCK_SIZE>>>(adjacency_matrix.num_rows, old_cc);
    cudaDeviceSynchronize();
    cudaCheckErrors("init old cc");
    
    size_t diff = 1;
    while(diff > 0) {
        cudaMemcpy(new_cc, old_cc, sizeof(size_t) * adjacency_matrix.num_rows, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        cudaCheckErrors("new_cc = old_cc");

        k_cc<<<NUM_BLOCKS(adjacency_matrix.num_rows), BLOCK_SIZE>>>(adjacency_matrix.num_rows, adjacency_matrix.row_ptr, adjacency_matrix.col_ptr, old_cc, new_cc);
        cudaDeviceSynchronize();
        cudaCheckErrors("k_cc");
        
        k_sub<<<NUM_BLOCKS(adjacency_matrix.num_rows), BLOCK_SIZE>>>(adjacency_matrix.num_rows, old_cc, new_cc);
        cudaDeviceSynchronize();
        cudaCheckErrors("k_sub");

        thrust::inclusive_scan(
            thrust::device,
            thrust::device_pointer_cast<size_t>(old_cc) + adjacency_matrix.num_rows,
            thrust::device_pointer_cast<size_t>(old_cc),
            thrust::device_pointer_cast<size_t>(old_cc)
        );

        cudaMemcpy(&diff, old_cc+(adjacency_matrix.num_rows-1), sizeof(size_t) * 1, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        cudaCheckErrors("get diff");
        std::swap(old_cc, new_cc);
        std::cout << "diff: " << diff << std::endl;
    }

    std::vector<size_t> cc(adjacency_matrix.num_rows);
    cudaMemcpy(cc.data(), old_cc, sizeof(size_t) * adjacency_matrix.num_rows, cudaMemcpyDefault);

    std::unordered_map<std::string, std::vector<uint64_t>> cc_map;
    for(size_t gpu_vertex_id = 0; gpu_vertex_id < adjacency_matrix.num_rows; ++gpu_vertex_id) {
        uint64_t cpu_vertex_id = boost::any_cast<uint64_t>(graph->access_vertices()[gpu_vertex_id]->id());
        cc_map[std::to_string(cc[gpu_vertex_id])].push_back(cpu_vertex_id);
    }

    std::unordered_map<std::string, boost::any> cc_output;
    cc_output[OUTPUT_COMPONENTS] = cc_map;

    cudaFree(old_cc);
    cudaFree(new_cc);

    return cc_output;
}

/*
    Computes A = (A ?= B)
*/
__global__ void k_sub(size_t N, size_t* A, size_t* B) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = index; i < N; i += stride) {
        A[i] = A[i] != B[i];
    }
}

/*
    Initialize the cc array.
*/
__global__ void k_cc_init(size_t N, size_t* cc) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = index; i < N; i += stride) {
        cc[i] = i;
    }
}


/*
    Connected Components kernel - performs a single iteration of the algorithm.
*/
__global__ void k_cc(size_t N, size_t* row_ptr, size_t* col_ptr, size_t* old_cc, size_t* new_cc) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = index; i < N; i += stride) {
        size_t row_start = row_ptr[i];
        size_t row_end = row_ptr[i+1];

        for(size_t j = row_start; j < row_end; ++j) {
            size_t col = col_ptr[j];
            size_t p = old_cc[col];
            if(p < new_cc[i]) new_cc[i] = p;
        }
    }
}
