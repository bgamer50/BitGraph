#ifndef GPU_TRAVERSAL_HELPER_H
#define GPU_TRAVERSAL_HELPER_H

// These are for kernels that heavily access global memory.
// Kernels that make better use of shared memory will have different params
#define BLOCK_SIZE 128
#define NUM_BLOCKS(N) ( (N + BLOCK_SIZE - 1) / BLOCK_SIZE )

#include <inttypes.h>
#include "traversal/Traverser.h"
#include "structure/GPUVertex.h"
#include "structure/GPUEdge.h"
#include "util/cuda_utils.h"

#include <cuda_runtime.h>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>

__global__ void k_quadvv_get_mem(int32_t* row_ptr, int32_t* T, int32_t* R, int N);
void t_quadvv_get_mem(int32_t* row_ptr, int32_t* T, int32_t* R, int N);
__global__ void k_quadvv_get_adj(int32_t* row_ptr, int32_t* col_ptr, int32_t* T, int32_t* ps, int32_t* O, int32_t* OO, int N);
__global__ void k_prefix_sum(int32_t* A, int32_t* B, int i, int N);
__global__ void k_pick_unique_trim(int32_t* V, int32_t* V_ptr, int32_t* U, int32_t* U_counts, size_t U_offset, size_t U_size);
__global__ void k_pick_unique(int32_t* A, size_t N, int32_t* U, int32_t* U_counts, int32_t U_offset, size_t U_size);

int32_t* to_gpu(TraverserSet& traversers);
void prefix_sum(int32_t** A_ptr, int N);
std::tuple<int32_t*, int32_t*, int> gpu_query_adjacency_v_to_v(sparse_matrix_device_t& M, int32_t* gpu_element_traversers, size_t N);
std::pair<std::pair<int32_t*, int32_t*>, int32_t*> gpu_query_adjacency_v_to_e(sparse_matrix_device_t& M, int32_t* gpu_element_traversers);

typedef struct gpu_traverser_info {
    int32_t* traversers;
    size_t num_traversers;
    TraverserSet original_traversers;
    std::vector<std::pair<int32_t*, size_t>> paths;
} gpu_traverser_info_t;

/**
    Copy data from a traversal over graph elements (Vertex,Edge)
    to the GPU.
**/
int32_t* to_gpu(TraverserSet& traversers) {
    const size_t sz = traversers.size();

    int32_t* gpu_traversers;
    cudaMalloc((void**) &gpu_traversers, sizeof(int32_t) * sz);
    cudaDeviceSynchronize();
    cudaCheckErrors("allocate traversers");

    std::vector<int32_t> trv(sz); 
    for(size_t k = 0; k < sz; ++k) {
        boost::any e = traversers[k].get();
        if(typeid(Vertex*) == e.type()) {
            trv[k] = static_cast<GPUVertex*>(boost::any_cast<Vertex*>(e))->gpu_vertex_id;
        } else if(typeid(Edge*) == e.type()) {
            throw std::runtime_error("GPU Traversal over edges currently unsupported due to GPU identifier requirements.");
            //trv[k] = static_cast<GPUEdge*>(boost::any_cast<Edge*>(e))->gpu_edge_id;
        } else {
            throw std::runtime_error("Type not supported for GPU traversal!");
        }
    } 
    
    cudaMemcpy(gpu_traversers, trv.data(), sizeof(int32_t) * sz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaCheckErrors("copy traversers to device");
    return gpu_traversers;
}

// (vertex id) -> (originating traverser)
/**
    M: The sparse adjacency matrix on the GPU
    gpu_element_traversers: The traversers as literal objects (an array of of Vertex ids.)
    N: The number of initial traversers.
**/
std::tuple<int32_t*, int32_t*, int32_t> gpu_query_adjacency_v_to_v(sparse_matrix_device_t& M, int32_t* gpu_element_traversers, size_t N) {
    int32_t* result;
    int32_t* output;
    int32_t* output_origin;
    
    cudaMalloc((void**) &result, sizeof(int32_t) * N);
    k_quadvv_get_mem<<<NUM_BLOCKS(N), BLOCK_SIZE>>>(M.row_ptr, gpu_element_traversers, result, N);
    //t_quadvv_get_mem(M.row_ptr, gpu_element_traversers, result, N);
    cudaDeviceSynchronize();
    cudaCheckErrors("k_quadvv_get_mem");

    prefix_sum(&result, N); // result now holds the prefix sums.

    // CPU needs to know ps information anyways, so we copy it and cudaMalloc the sum
    int32_t N_prime; // = result[N-1]; # of output traversers
    cudaMemcpy(&N_prime, &result[N-1], sizeof(int32_t) * 1, cudaMemcpyDeviceToHost);
    
    cudaMalloc((void**) &output, sizeof(int32_t) * N_prime);
    cudaMalloc((void**) &output_origin, sizeof(int32_t) * N_prime);

    //std::cout << "num traversers: " << N << std::endl;
    //std::cout << "num blocks: " << NUM_BLOCKS(N) << std::endl;

    // Then we run a kernel that actually spits out the column #s (a.k.a. adjacent vertices in the out-direction, or in-direction if this matrix has been transposed)
    k_quadvv_get_adj<<<NUM_BLOCKS(N), BLOCK_SIZE>>>(M.row_ptr, M.col_ptr, gpu_element_traversers, result, output, output_origin, N);
    cudaDeviceSynchronize();
    cudaCheckErrors("k_quadvv_get_adj");
    
    return std::make_tuple(output, output_origin, N_prime);
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

struct plus_op : public thrust::unary_function<thrust::tuple<int32_t,int32_t>,int32_t> {
    __device__ int32_t operator()(thrust::tuple<int32_t, int32_t> t) const {
        return thrust::get<0>(t) + thrust::get<1>(t);
    }
};

struct minus_op : public thrust::unary_function<thrust::tuple<int32_t,int32_t>,int32_t> {
    __device__ int32_t operator()(thrust::tuple<int32_t, int32_t> t) const {
        return thrust::get<0>(t) - thrust::get<1>(t);
    }
};

void t_quadvv_get_mem(int32_t* row_ptr, int32_t* T, int32_t* R, int N) {
    thrust::device_ptr<int32_t> row_dptr = thrust::device_pointer_cast(row_ptr);
    thrust::device_ptr<int32_t> T_dptr = thrust::device_pointer_cast(T);

    thrust::constant_iterator<int32_t> single = thrust::make_constant_iterator<int32_t>(1);

    auto T_plus_one = thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(T_dptr, single)
            ),
            plus_op()
    );

    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_permutation_iterator(
                row_dptr,
                T_plus_one
            ),
            thrust::make_permutation_iterator(
                row_dptr,
                T_dptr
            )
        )
    );

    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_permutation_iterator(
                row_dptr,
                T_plus_one + N
            ),
            thrust::make_permutation_iterator(
                row_dptr,
                T_dptr + N
            )
        )
    );

    thrust::copy(thrust::make_transform_iterator(zip_begin, minus_op()), thrust::make_transform_iterator(zip_end, minus_op()), thrust::device_pointer_cast(R));
}

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
            OO[output_index] = i;
            ++output_index;
        }
    }
}

// gridDim.x * blockDim.x = # traversers
// blockDim.y = max degree
/*
__global__ void k_quadvv_get_adj_v2(int32_t* row_ptr, int32_t* col_ptr, int32_t* T, int32_t* ps, int32_t* O, int32_t* OO, int N, int M, int nnz) {
    extern __shared__ int32_t s[];

    int index = gridDim.x * blockIdx.x + blockDim.x * threadIdx.x + threadIdx.y;
    int vertex = T[i];
    int output_index = i==0 ? 0 : ps[i - 1]; // TODO might be able to put this in shared memory

    s[0] = row_ptr[vertex];
    s[1] = row_ptr[vertex + 1];

    
}
*/

void t_quadvv_get_adj(int32_t* row_ptr, int32_t* col_ptr, int32_t* T, int32_t* ps, int32_t* O, int32_t* OO, int N, int M, int nnz) {

}

void prefix_sum(int32_t** A_ptr, int N) {
    //thrust::device_ptr<int32_t> A_dptr = thrust::device_pointer_cast(*A_ptr);
    //thrust::inclusive_scan(A_dptr, A_dptr+N, A_dptr);
    

    int32_t* A = *A_ptr;
    int32_t* temp;
    cudaMalloc((void**) &temp, sizeof(int32_t) * N);
    
    for(int i = 1; i < N; i *= 2) {
        k_prefix_sum<<<NUM_BLOCKS(N), BLOCK_SIZE>>>(A, temp, i, N);
        cudaDeviceSynchronize();
        cudaCheckErrors("k_prefix_sum");
        std::swap(A, temp);
    }

    cudaFree(temp);
    *A_ptr = A;
    
    
}

/**
    Helper method for prefix_sum (device kernel)
**/
__global__ void k_prefix_sum(int32_t* A, int32_t* B, int i, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < N; j += stride) {
        if(j < i) B[j] = A[j];
        else B[j] = A[j] + A[j-i];
    }
}

// (outV id, inV id) -> (originating traverser)
std::pair<std::pair<int32_t*, int32_t*>, int32_t*> gpu_query_adjacency_v_to_e(sparse_matrix_device_t& M, int32_t* gpu_element_traversers) {
    // TODO don't bother supporting this until EdgeVertexStep is implemented in Gremlin++
    throw std::runtime_error("Cannot currently query adjacency from Vertex to Edge!");
}

/**
    For each possible integer from the set U (U_start through U_end, inclusive), an
    array is returned whose values are the picked element indices from A, and other is returned
    with the values themselves.  For instance, take A=[1,2,3,1] and U = [-3..3].  The returned 
    arrays V_ptr=[0,1,2] and V=[1,2,3] points to nonduplicate
    elements 0, 1, and 2 from A.

    V: The array of actual deduplicated values.
    V_ptr: The array pointing to each deduplicated value's origin
    V_size: The length of V and V_ptr (# of deduplicated elements).
**/
std::tuple<int32_t*, int32_t*, int32_t> pick_unique(int32_t* A, size_t N, int32_t U_start, int32_t U_end) {
    size_t U_size = U_end - U_start + 1;

    int32_t* U;
    int32_t* U_counts;
    cudaMalloc((void**) &U, sizeof(int32_t) * U_size);
    cudaMalloc((void**) &U_counts, sizeof(int32_t) * U_size);
    cudaDeviceSynchronize();
    cudaCheckErrors("Allocate arrays U and U_counts in pick_unique");


    k_pick_unique<<<NUM_BLOCKS(U_size), BLOCK_SIZE>>>(A, N, U, U_counts, U_start, U_size);
    cudaDeviceSynchronize();
    cudaCheckErrors("Call k_pick_unique device kernel");
    
    prefix_sum(&U_counts, U_size);
    int32_t V_size;
    cudaMemcpy(&V_size, U_counts + (U_size - 1), sizeof(int32_t) * 1, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckErrors("Get values array size in pick_unique");

    int32_t* V;
    int32_t* V_ptr;
    cudaMalloc((void**) &V, sizeof(int32_t) * V_size);
    cudaMalloc((void**) &V_ptr, sizeof(int32_t) * V_size);
    cudaDeviceSynchronize();
    cudaCheckErrors("Allocate value arras V, V_ptr in pick_unique");

    k_pick_unique_trim<<<NUM_BLOCKS(U_size), BLOCK_SIZE>>>(V, V_ptr, U, U_counts, U_start, U_size);
    cudaDeviceSynchronize();
    cudaCheckErrors("Call k_pick_unique_trim device kernel");

    cudaFree(U);
    cudaFree(U_counts);
    cudaDeviceSynchronize();
    cudaCheckErrors("Free arrays U, U_counts");

    return std::make_tuple(V, V_ptr, V_size);
}

__global__ void k_pick_unique_trim(int32_t* V, int32_t* V_ptr, int32_t* U, int32_t* U_counts, size_t U_offset, size_t U_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < U_size; j += stride) {
        int Uj = U[j];
        if(Uj >= 0) {
            int pos_V = j==0 ? 0 : U_counts[j-1];
            V_ptr[pos_V] = Uj;
            V[pos_V] = j + U_offset;
        }
    }
}

/**
    Helper method for pick_unique (device kernel)
    A: original array
    N: size of A
    U: set of possible elements mapped to their first indices in A if present
    U_counts: count for each element in U
    U_offset: first value of U
    U_size: size of U
**/
__global__ void k_pick_unique(int32_t* A, size_t N, int32_t* U, int32_t* U_counts, int32_t U_offset, size_t U_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // note: N is not the end of this loop - in this case it's U_size since each executor is assigned to an element of U.
    for (int j = index; j < U_size; j += stride) {
        int U_val = j + U_offset;
        U[j] = -1;
        U_counts[j] = 0;
        for(int k = 0; k < N; ++k) if(A[k] == U_val) {
            U[j] = k;
            U_counts[j] = 1;
            break;
        }
    }
}

/**
   Collapses a path down into a single output origin array.
   traverser_info: the traverser info (contains path & other info)
**/
std::vector<int32_t> collapse_path(gpu_traverser_info_t& traverser_info, bool free_memory) {
    size_t OO_size = traverser_info.paths.back().second;

    int32_t* OO = traverser_info.paths.back().first;

    thrust::device_ptr<int32_t> d_ptr_OO = thrust::device_pointer_cast(OO);

    for(auto it = traverser_info.paths.rbegin() + 1; it != traverser_info.paths.rend(); ++it) {
        thrust::device_ptr<int32_t> d_ptr_previous_traversers = thrust::device_pointer_cast(it->first);
        size_t num_previous_traversers = it->second;

        thrust::copy(
            thrust::make_permutation_iterator(d_ptr_previous_traversers, d_ptr_OO),
            thrust::make_permutation_iterator(d_ptr_previous_traversers, d_ptr_OO + OO_size),
            d_ptr_OO
        );

        if(free_memory) cudaFree(it->first);
    }

    std::vector<int32_t> returned_oo_cpu(OO_size);
    cudaMemcpy(returned_oo_cpu.data(), OO, sizeof(int32_t) * OO_size, cudaMemcpyDeviceToHost);
    if(free_memory) cudaFree(OO);
    return returned_oo_cpu;
}

#endif