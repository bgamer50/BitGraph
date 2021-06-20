#ifndef GPU_TRAVERSAL_HELPER_H
#define GPU_TRAVERSAL_HELPER_H

#include <inttypes.h>
#include "traversal/Traverser.h"

#include <cuda_runtime.h>

__global__ void k_quadvv_get_adj(int32_t* row_ptr, int32_t* col_ptr, int32_t* T, int32_t* ps, int32_t* O, int32_t* OO, int N);
__global__ void k_quadvv_get_mem(int32_t* row_ptr, int32_t* T, int32_t* R, int N);
__global__ void k_prefix_sum(int32_t* A, int32_t* B, int i, int N);
int32_t* to_gpu(TraverserSet& traversers);
void prefix_sum(int32_t** A_ptr, int N);
std::tuple<int32_t*, int32_t*, int> gpu_query_adjacency_v_to_v(sparse_matrix_device_t& M, int32_t* gpu_element_traversers, size_t N);
std::pair<std::pair<int32_t*, int32_t*>, int32_t*> gpu_query_adjacency_v_to_e(sparse_matrix_device_t& M, int32_t* gpu_element_traversers);

/**
    Copy data from a traversal over graph elements (Vertex,Edge)
    to the GPU.
**/
int32_t* to_gpu(TraverserSet& traversers) {
    const size_t sz = traversers.size();

    int32_t* gpu_traversers;
    cudaMalloc((void**) &gpu_traversers, sizeof(int32_t) * sz);

    std::vector<int32_t> trv(sz); 
    for(size_t k = 0; k < sz; ++k) {
        boost::any e = traversers[k].get();
        if(typeid(Vertex*) == e.type()) {
            trv[k] = static_cast<GPUReferenceVertex*>(boost::any_cast<Vertex*>(traversers[k]))->gpu_vertex_id;
        } else if(typeid(Edge*) == e.type()) {
            trv[k] = static_cast<GPUReferenceEdge*>(boost::any_cast<Edge*>(traversers[k]))->gpu_edge_id;
        } else {
            throw std::runtime_error("Type not supported for GPU traversal!");
        }
    } 
    
    cudaMemcpy(gpu_traversers, trv.data(), sizeof(int32_t) * sz, cudaMemcpyHostToDevice); // blocking is implied here
    return gpu_traversers;
}

// (vertex id) -> (originating traverser)
/**
    M: The sparse adjacency matrix on the GPU
    gpu_element_traversers: The traversers as literal objects (an array of of Vertex ids.)
    N: The number of initial traversers.
**/
std::tuple<int32_t*, int32_t*, int> gpu_query_adjacency_v_to_v(sparse_matrix_device_t& M, int32_t* gpu_element_traversers, size_t N) {
    int32_t* result;
    int32_t* output;
    int32_t* output_origin;
    
    cudaMalloc((void**) &result, sizeof(int32_t) * N);
    k_quadvv_get_mem(M.row_ptr, gpu_element_traversers, result, N);
    prefix_sum(&result, N); // result now holds the prefix sums.

    // CPU needs to know ps information anyways, so we copy it and cudaMalloc the sum
    const int N_prime = result[N-1]; // # of output traversers
    cudaMalloc((void**) &output, sizeof(int32_t) * N_prime);
    cudaMalloc((void**) &output_origin, sizeof(int32_t) * N_prime);

    // Then we run a kernel that actually spits out the column #s (a.k.a. adjacent vertices in the out-direction)
    // TODO this doesn't work for the in-direction
    k_quadvv_get_adj(M.row_ptr, M.col_ptr, gpu_element_traversers, result, output, output_origin, N);
    return std::make_tuple(output, output_origin, N_prime);
}

void prefix_sum(int32_t** A_ptr, int N) {
    int32_t* A = *A_ptr;
    int32_t* temp;
    cudaMalloc((void**) &temp, sizeof(int32_t) * N);
    
    for(int i = 1; i < N; i *= 2) {
        k_prefix_sum(A, temp, i, N);
        std::swap(A, temp);
        cudaDeviceSynchronize();
    }

    cudaFree(temp);
    *A_ptr = A;
}

// (outV id, inV id) -> (originating traverser)
std::pair<std::pair<int32_t*, int32_t*>, int32_t*> gpu_query_adjacency_v_to_e(sparse_matrix_device_t& M, int32_t* gpu_element_traversers) {
    // TODO don't bother supporting this until EdgeVertexStep is implemented in Gremlin++
    throw std::runtime_error("Cannot currently query adjacency from Vertex to Edge!");
}

#endif