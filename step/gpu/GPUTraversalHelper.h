#pragma once

// These are for kernels that heavily access global memory.
// Kernels that make better use of shared memory will have different params
#define BLOCK_SIZE 128
#define NUM_BLOCKS(N) ( (N + BLOCK_SIZE - 1) / BLOCK_SIZE )

#include <inttypes.h>
#include "traversal/Traverser.h"
#include "traversal/Comparison.h"
#include "structure/GPUVertex.h"
#include "structure/GPUEdge.h"
#include "util/cuda_utils.h"

#include <cuda_runtime.h>

#include "structure/memory/ThrustUtils.h"

__global__ void k_quadvv_get_mem(size_t* row_ptr, size_t* T, size_t* R, int N);
void t_quadvv_get_mem(size_t* row_ptr, size_t* T, size_t* R, int N);
__global__ void k_quadvv_get_adj(size_t* row_ptr, size_t* col_ptr, size_t* T, size_t* ps, size_t* O, size_t* OO, int N);
__global__ void k_prefix_sum(size_t* A, size_t* B, int i, int N);

void prefix_sum(size_t** A_ptr, int N);
std::tuple<size_t*, size_t*, size_t> gpu_query_adjacency_v_to_v(bitgraph::matrix::sparse_matrix_device& M, size_t* gpu_element_traversers, size_t N);
std::pair<std::pair<size_t*, size_t*>, size_t*> gpu_query_adjacency_v_to_e(bitgraph::matrix::sparse_matrix_device& M, size_t* gpu_element_traversers);

// TODO write a function for properly updating this data structure
typedef struct gpu_traverser_info {
    void* traversers;
    size_t num_traversers;
    gremlinxx::comparison::C traverser_dtype;
    TraverserSet original_traversers;
    std::vector<std::pair<size_t*, size_t>> paths;
} gpu_traverser_info_t;

enum reduction_type{MIN, MAX, SUM, MUL};

/**
    Copy data from a traversal over graph elements (Vertex,Edge)
    to the GPU.
**/
template<typename T>
void* to_gpu(TraverserSet& traversers) {
    const size_t sz = traversers.size();
    std::cout << "# traversers: " << sz << std::endl;
    std::cout << "trv size: " << sizeof(T) << std::endl;

    T* gpu_traversers;
    cudaMalloc((void**) &gpu_traversers, sizeof(T) * sz);
    cudaDeviceSynchronize();
    cudaCheckErrors("allocate traversers");

    std::vector<T> trv(sz); 
    for(size_t k = 0; k < sz; ++k) {
        try {
            trv[k] = boost::any_cast<T>(traversers[k].get());
        } catch(boost::bad_any_cast& ex) {
            std::stringstream ss;
            ss << ex.what() << std::endl;
            ss << "unexpected type: " << boost::core::demangled_name(traversers[k].get().type()) << std::endl;
            throw std::runtime_error(ss.str());
        }
    } 
    
    cudaMemcpy(gpu_traversers, trv.data(), sizeof(size_t) * sz, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaCheckErrors("copy traversers to device");
    return (void*)gpu_traversers;
}

template<>
void* to_gpu<Vertex*>(TraverserSet& traversers) {
    const size_t sz = traversers.size();
    std::cout << "# traversers: " << sz << std::endl;

    size_t* gpu_traversers;
    cudaMalloc((void**) &gpu_traversers, sizeof(size_t) * sz);
    cudaDeviceSynchronize();
    cudaCheckErrors("allocate traversers");

    std::vector<size_t> trv(sz); 
    for(size_t k = 0; k < sz; ++k) {
        try {
            Vertex* v = boost::any_cast<Vertex*>(traversers[k].get());
            GPUVertex* gv = static_cast<GPUVertex*>(v);
            trv[k] = gv->gpu_vertex_id;
        } catch(boost::bad_any_cast& ex) {
            std::stringstream ss;
            ss << ex.what() << std::endl;
            ss << "expected a Vertex" << std::endl;
            ss << "but got: " << boost::core::demangled_name(traversers[k].get().type()) << std::endl;
            throw std::runtime_error(ss.str());
        }
    } 
    
    cudaMemcpy(gpu_traversers, trv.data(), sizeof(size_t) * sz, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaCheckErrors("copy traversers to device");
    return (void*)gpu_traversers;
}

template
void* to_gpu<uint64_t>(TraverserSet& traversers);
template
void* to_gpu<uint32_t>(TraverserSet& traversers);
template
void* to_gpu<uint8_t>(TraverserSet& traversers);
template
void* to_gpu<int64_t>(TraverserSet& traversers);
template
void* to_gpu<int32_t>(TraverserSet& traversers);
template
void* to_gpu<int8_t>(TraverserSet& traversers);
template
void* to_gpu<float>(TraverserSet& traversers);
template
void* to_gpu<double>(TraverserSet& traversers);

void* C_TO_GPU(gremlinxx::comparison::C c, TraverserSet& traversers) {
    switch(c) {
        case gremlinxx::comparison::C::UINT64:
            return to_gpu<uint64_t>(traversers);
        case gremlinxx::comparison::C::UINT32:
            return to_gpu<uint32_t>(traversers);
        case gremlinxx::comparison::C::UINT8:
            return to_gpu<uint8_t>(traversers);
        case gremlinxx::comparison::C::INT64:
            return to_gpu<int64_t>(traversers);
        case gremlinxx::comparison::C::INT32:
            return to_gpu<int32_t>(traversers);
        case gremlinxx::comparison::C::INT8:
            return to_gpu<int8_t>(traversers);
        case gremlinxx::comparison::C::FLOAT64:
            return to_gpu<double>(traversers);
        case gremlinxx::comparison::C::FLOAT32:
            return to_gpu<float>(traversers);
        case gremlinxx::comparison::C::VERTEX:
            return to_gpu<Vertex*>(traversers);
    }

    throw std::runtime_error("Illegal type provided");
}

// (vertex id) -> (originating traverser)
/**
    M: The sparse adjacency matrix on the GPU
    gpu_element_traversers: The traversers as literal objects (an array of of Vertex ids.)
    N: The number of initial traversers.
**/
std::tuple<size_t*, size_t*, size_t> gpu_query_adjacency_v_to_v(bitgraph::matrix::sparse_matrix_device& M, size_t* gpu_element_traversers, size_t N) {
    size_t* result;
    size_t* output;
    size_t* output_origin;
    
    cudaMalloc((void**) &result, sizeof(size_t) * N);
    k_quadvv_get_mem<<<NUM_BLOCKS(N), BLOCK_SIZE>>>(M.row_ptr, gpu_element_traversers, result, N);
    //t_quadvv_get_mem(M.row_ptr, gpu_element_traversers, result, N);
    cudaDeviceSynchronize();
    cudaCheckErrors("k_quadvv_get_mem");

    prefix_sum(&result, N); // result now holds the prefix sums.

    // CPU needs to know ps information anyways, so we copy it and cudaMalloc the sum
    size_t N_prime; // = result[N-1]; # of output traversers
    cudaMemcpy(&N_prime, &result[N-1], sizeof(size_t) * 1, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaCheckErrors("copy result to host");
    
    cudaMalloc((void**) &output, sizeof(size_t) * N_prime);
    cudaMalloc((void**) &output_origin, sizeof(size_t) * N_prime);
    cudaDeviceSynchronize();
    cudaCheckErrors("allocate output and output origin");

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
__global__ void k_quadvv_get_mem(size_t* row_ptr, size_t* T, size_t* R, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < N; i += stride) {
        size_t vertex = T[i];
        size_t start = row_ptr[vertex];
        size_t end = row_ptr[vertex + 1];
        
        size_t v_out_count = end - start;
        R[i] = v_out_count;
    }
}

void t_quadvv_get_mem(size_t* row_ptr, size_t* T, size_t* R, int N) {
    thrust::device_ptr<size_t> row_dptr = thrust::device_pointer_cast(row_ptr);
    thrust::device_ptr<size_t> T_dptr = thrust::device_pointer_cast(T);

    thrust::constant_iterator<size_t> single = thrust::make_constant_iterator<size_t>(1);

    auto T_plus_one = thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(T_dptr, single)
            ),
            bitgraph::memory::plus_op()
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

    thrust::copy(thrust::make_transform_iterator(zip_begin, bitgraph::memory::minus_op()), thrust::make_transform_iterator(zip_end, bitgraph::memory::minus_op()), thrust::device_pointer_cast(R));
}

/**
    M: (row_ptr, col_ptr) adjacency matrix
    T: intial traversers
    ps: prefix summed result array (indicates index of start for each traverser; last element is total size)
    O: The output array of traversers
    OO: The output origin array (which original traverser did the new traverser originate from)
    N: The original number of traversers
**/
__global__ void k_quadvv_get_adj(size_t* row_ptr, size_t* col_ptr, size_t* T, size_t* ps, size_t* O, size_t* OO, int N) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = index; i < N; i += stride) {
        size_t vertex = T[i];
        size_t output_index = i==0 ? 0 : ps[i - 1];

        size_t start = row_ptr[vertex];
        size_t end = row_ptr[vertex + 1];
        
        for(size_t j = start; j < end; ++j) {
            O[output_index] = col_ptr[j];
            OO[output_index] = i;
            ++output_index;
        }
    }
}

// This will probably eventually replace k_quadvv_get_adj
void t_quadvv_get_adj(size_t* row_ptr, size_t* col_ptr, size_t* T, size_t* ps, size_t* O, size_t* OO, int N, int M, int nnz) {

}

void prefix_sum(size_t** A_ptr, int N) {
    //thrust::device_ptr<size_t> A_dptr = thrust::device_pointer_cast(*A_ptr);
    //thrust::inclusive_scan(A_dptr, A_dptr+N, A_dptr);
    

    size_t* A = *A_ptr;
    size_t* temp;
    cudaMalloc((void**) &temp, sizeof(size_t) * N);
    
    for(size_t i = 1; i < N; i *= 2) {
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
__global__ void k_prefix_sum(size_t* A, size_t* B, size_t i, size_t N) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t j = index; j < N; j += stride) {
        if(j < i) B[j] = A[j];
        else B[j] = A[j] + A[j-i];
    }
}

// (outV id, inV id) -> (originating traverser)
std::pair<std::pair<size_t*, size_t*>, size_t*> gpu_query_adjacency_v_to_e(bitgraph::matrix::sparse_matrix_device& M, size_t* gpu_element_traversers) {
    // TODO don't bother supporting this until EdgeVertexStep is implemented in Gremlin++
    throw std::runtime_error("Cannot currently query adjacency from Vertex to Edge!");
}

/**
    Removes duplicates from A (modifies the original device array).
    Returns the original indices of the unique elements and the
    number of unique elements (size of index array and new size of A).

    This method is NOT stable (may choose unique indices that are not
    the first occurrence of that value).

    V_ptr: The array pointing to each deduplicated value's origin
    V_size: The length of V and V_ptr (# of deduplicated elements).
**/
std::tuple<size_t*, size_t> pick_unique(size_t* A, size_t N) {
    thrust::device_ptr<size_t> A_tptr = thrust::device_pointer_cast(A);

    size_t* V_ptr;
    cudaMalloc(&V_ptr, sizeof(size_t) * N);
    cudaDeviceSynchronize();
    cudaCheckErrors("allocate V_ptr");
    thrust::device_ptr<size_t> V_ptr_tptr = thrust::device_pointer_cast(V_ptr);

    auto seq_it = thrust::make_counting_iterator((size_t)0);
    thrust::copy(
        seq_it,
        seq_it + N,
        V_ptr_tptr
    );

    // Sort the kv pairs since unique_by_key will only remove
    // consecutive unique elements.
    thrust::sort_by_key(
        A_tptr,
        A_tptr + N,
        V_ptr_tptr
    );

    thrust::device_ptr<size_t> A_end;
    thrust::device_ptr<size_t> V_ptr_end;
    thrust::tie(A_end, V_ptr_end) = thrust::unique_by_key(
        A_tptr,
        A_tptr + N,
        V_ptr_tptr
    );

    size_t V_size = V_ptr_end - V_ptr_tptr;
    return std::make_tuple(V_ptr, V_size);
}

/**
   Collapses a path down into a single output origin array.
   traverser_info: the traverser info (contains path & other info)
**/
std::vector<size_t> collapse_path(gpu_traverser_info_t& traverser_info, bool free_memory) {
    size_t OO_size = traverser_info.paths.back().second;

    size_t* OO = traverser_info.paths.back().first;

    thrust::device_ptr<size_t> d_ptr_OO = thrust::device_pointer_cast(OO);

    for(auto it = traverser_info.paths.rbegin() + 1; it != traverser_info.paths.rend(); ++it) {
        thrust::device_ptr<size_t> d_ptr_previous_traversers = thrust::device_pointer_cast(it->first);
        size_t num_previous_traversers = it->second;

        thrust::copy(
            thrust::make_permutation_iterator(d_ptr_previous_traversers, d_ptr_OO),
            thrust::make_permutation_iterator(d_ptr_previous_traversers, d_ptr_OO + OO_size),
            d_ptr_OO
        );

        if(free_memory) cudaFree(it->first);
    }

    std::vector<size_t> returned_oo_cpu(OO_size);
    cudaMemcpy(returned_oo_cpu.data(), OO, sizeof(size_t) * OO_size, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaCheckErrors("Copy output origin to CPU");

    if(free_memory) cudaFree(OO);
    cudaDeviceSynchronize();
    cudaCheckErrors("free output origin");
    return returned_oo_cpu;
}

template<typename T>
void retrieve_new_traversers(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info) {
    std::vector<size_t> originating_traversers = collapse_path(traverser_info, true); // don't handle path info at the moment

    std::vector<T> new_traversers_raw(traverser_info.num_traversers);
    cudaMemcpy(new_traversers_raw.data(), (T*)traverser_info.traversers, sizeof(T) * traverser_info.num_traversers, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaCheckErrors("Copy traversers to CPU");
    
    size_t old_size = output_traversers.size();
    output_traversers.resize(old_size + traverser_info.num_traversers);
    for(int k = 0; k < traverser_info.num_traversers; ++k) {
        Traverser& originating_traverser = traverser_info.original_traversers[originating_traversers[k]];
        
        output_traversers[old_size + k].replace_data(new_traversers_raw[k]);
        auto& se = originating_traverser.get_side_effects();
        output_traversers[old_size + k].get_side_effects().insert(se.begin(), se.end());
    }

}

template<>
void retrieve_new_traversers<Vertex*>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info) {
    std::vector<size_t> originating_traversers = collapse_path(traverser_info, true); // don't handle path info at the moment

    std::vector<size_t> new_traversers_raw(traverser_info.num_traversers);
    cudaMemcpy(
        new_traversers_raw.data(),
        static_cast<size_t*>(traverser_info.traversers),
        sizeof(size_t) * traverser_info.num_traversers,
        cudaMemcpyDefault
    );
    cudaDeviceSynchronize();
    cudaCheckErrors("Copy vertex traversers to CPU");
    
    GPUGraph* gpu_graph = static_cast<GPUGraph*>(parent_traversal->getGraph());
    size_t old_size = output_traversers.size();
    output_traversers.resize(old_size + traverser_info.num_traversers);
    for(int k = 0; k < traverser_info.num_traversers; ++k) {
        Vertex* v = static_cast<Vertex*>(gpu_graph->access_vertices()[new_traversers_raw[k]]);
        Traverser& originating_traverser = traverser_info.original_traversers[originating_traversers[k]];
        
        output_traversers[old_size + k].replace_data(v);
        auto& se = originating_traverser.get_side_effects();
        output_traversers[old_size + k].get_side_effects().insert(se.begin(), se.end());
    }

}

template
void retrieve_new_traversers<uint64_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<uint32_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<uint8_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<int64_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<int32_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<int8_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<double>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
void retrieve_new_traversers<float>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);

void C_RETRIEVE_NEW_TRAVERSERS(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info) {
    switch(traverser_info.traverser_dtype) {
        case gremlinxx::comparison::C::UINT64:
            return retrieve_new_traversers<uint64_t>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::UINT32:
            return retrieve_new_traversers<uint32_t>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::UINT8:
            return retrieve_new_traversers<uint8_t>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::INT64:
            return retrieve_new_traversers<int64_t>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::INT32:
            return retrieve_new_traversers<int32_t>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::INT8:
            return retrieve_new_traversers<int8_t>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::FLOAT64:
            return retrieve_new_traversers<double>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::FLOAT32:
            return retrieve_new_traversers<float>(parent_traversal, output_traversers, traverser_info);
        case gremlinxx::comparison::C::VERTEX:
            return retrieve_new_traversers<Vertex*>(parent_traversal, output_traversers, traverser_info);
    }

    throw std::runtime_error("Illegal type provided");
}
