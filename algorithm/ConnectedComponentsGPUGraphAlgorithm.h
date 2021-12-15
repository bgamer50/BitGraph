#ifndef CONNECTED_COMPONENTS_GPU_GRAPH_ALGORITHM_H
#define CONNECTED_COMPONENTS_GPU_GRAPH_ALGORITHM_H

#include "algorithm/GPUGraphAlgorithm.h"
#include "structure/Direction.h"

__global__ void k_cc(int N, int32_t* row_ptr, int32_t* col_ptr, int32_t* old_cc, int32_t* new_cc);
__global__ void k_sub(int N, int32_t* A, int32_t* B);
__global__ void k_cc_init(int N, int32_t* cc);

class GPUGraph;

class ConnectedComponentsGPUGraphAlgorithm : public GPUGraphAlgorithm {
    private:
        Direction direction = BOTH;

    public:
        static const inline std::string OPTION_DIRECTION = "DIRECTION";
        static const inline std::string OUTPUT_COMPONENTS = "COMPONENTS";

        virtual std::unordered_map<std::string, boost::any> exec(GPUGraph* graph);

        virtual GPUGraphAlgorithm* option(std::string opt, boost::any value) {
            if(opt == OPTION_DIRECTION) {
                this->direction = boost::any_cast<Direction>(value);
            } else {
                throw std::runtime_error("Invalid option " + opt);
            }

            return this;
        }
};

#include "structure/GPUGraph.h"
#include "step/gpu/GPUTraversalHelper.h" // for prefix_sum

std::unordered_map<std::string, boost::any> ConnectedComponentsGPUGraphAlgorithm::exec(GPUGraph* graph) {
    sparse_matrix_device_t adjacency_matrix = graph->get_adjacency_matrix(this->direction);

    int32_t* new_cc = nullptr; cudaMalloc((void**) &new_cc, sizeof(int32_t) * adjacency_matrix.num_rows);
    cudaDeviceSynchronize();
    cudaCheckErrors("malloc new cc");
    int32_t* old_cc = nullptr; cudaMalloc((void**) &old_cc, sizeof(int32_t) * adjacency_matrix.num_rows);
    cudaDeviceSynchronize();
    cudaCheckErrors("malloc old cc");

    k_cc_init<<<NUM_BLOCKS(adjacency_matrix.num_rows), BLOCK_SIZE>>>(adjacency_matrix.num_rows, old_cc);
    cudaDeviceSynchronize();
    cudaCheckErrors("init old cc");
    
    size_t num_edges = adjacency_matrix.nnz;
    int32_t diff = 1;
    while(diff > 0) {
        cudaMemcpy(new_cc, old_cc, sizeof(int32_t) * adjacency_matrix.num_rows, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaCheckErrors("new_cc = old_cc");

        k_cc<<<NUM_BLOCKS(adjacency_matrix.num_rows), BLOCK_SIZE>>>(adjacency_matrix.num_rows, adjacency_matrix.row_ptr, adjacency_matrix.col_ptr, old_cc, new_cc);
        cudaDeviceSynchronize();
        cudaCheckErrors("k_cc");
        
        k_sub<<<NUM_BLOCKS(adjacency_matrix.num_rows), BLOCK_SIZE>>>(adjacency_matrix.num_rows, old_cc, new_cc);
        cudaDeviceSynchronize();
        cudaCheckErrors("k_sub");

        prefix_sum(&old_cc, adjacency_matrix.num_rows);

        cudaMemcpy(&diff, old_cc+(adjacency_matrix.num_rows-1), sizeof(int32_t) * 1, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaCheckErrors("get diff");
        std::swap(old_cc, new_cc);
        std::cout << "diff: " << diff << std::endl;
    }

    std::vector<int32_t> cc(adjacency_matrix.num_rows);
    cudaMemcpy(cc.data(), old_cc, sizeof(int32_t) * adjacency_matrix.num_rows, cudaMemcpyDeviceToHost);

    std::unordered_map<std::string, std::vector<uint64_t>> cc_map;
    for(int32_t gpu_vertex_id = 0; gpu_vertex_id < adjacency_matrix.num_rows; ++gpu_vertex_id) {
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
__global__ void k_sub(int N, int32_t* A, int32_t* B) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        A[i] = A[i] != B[i];
    }
}

/*
    Initialize the cc array.
*/
__global__ void k_cc_init(int N, int32_t* cc) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        cc[i] = i;
    }
}


/*
    Connected Components kernel - performs a single iteration of the algorithm.
*/
__global__ void k_cc(int N, int32_t* row_ptr, int32_t* col_ptr, int32_t* old_cc, int32_t* new_cc) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i+1];

        for(int j = row_start; j < row_end; ++j) {
            int col = col_ptr[j];
            int p = old_cc[col];
            if(p < new_cc[i]) new_cc[i] = p;
        }
    }
}

#endif