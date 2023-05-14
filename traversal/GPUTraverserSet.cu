#include "traversal/GPUTraverserSet.cuh"

#include "util/cuda_utils.cuh"

#include <cuda_runtime.h>
#include "structure/GPUVertex.cuh"
#include "structure/GPUEdge.cuh"
#include "structure/GPUGraph.cuh"
#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace traversal {
        
        TraverserSet GPUTraverserSet::to_cpu_traversers(GraphTraversal* parent_traversal, StringIndex* string_index) {
            auto traverser_data_anys = bitgraph::memory::vector_to_anys(
                this->traverser_data,
                string_index
            );

            std::unordered_map<std::string, std::vector<boost::any>> side_effects_anys;
            for(auto e = this->side_effects.begin(); e != this->side_effects.end(); ++e) {
                side_effects_anys[e->first] = bitgraph::memory::vector_to_anys(
                    e->second,
                    string_index
                );
            }

            if(this->persist_paths) {
                throw std::runtime_error("persist paths currently unsupported!");
            }

            TraverserSet new_traversers(this->size());
            for(size_t k = 0; k < new_traversers.size(); ++k) {
                // Fill traverser data
                new_traversers[k].replace_data(traverser_data_anys[k]);

                // Fill traverser side effects
                for(auto se = side_effects_anys.begin(); se != side_effects_anys.end(); ++se) {
                    auto se_value = se->second[k];
                    new_traversers[k].get_side_effects()[se->first] = se_value;
                }

                // Fill traverser paths
                if(this->persist_paths) {
                    throw std::runtime_error("persist paths currently unsupported!");
                }
            }
        }

        void GPUTraverserSet::from_cpu_traversers(TraverserSet& cpu_traversers, StringIndex* string_index) {
            this->clear();

            
        }

        void GPUTraverserSet::clear() {
            this->traverser_data.clear();
            this->paths.clear();
            this->side_effects.clear();
        }

    }
}

/**
    Copy data from a traversal over graph elements (Vertex,Edge)
    to the GPU.
**/
template<typename T>
void* to_gpu(TraverserSet& traversers) {
    const size_t sz = traversers.size();

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

        if(free_memory) {
            cudaFree(it->first);
            cudaDeviceSynchronize();
            cudaCheckErrors("free path data");
        }
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
TraverserSet retrieve_new_traversers<uint64_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<uint32_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<uint8_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<int64_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<int32_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<int8_t>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<double>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
template
TraverserSet retrieve_new_traversers<float>(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);

TraverserSet gpu_traversers_to_cpu_traversers(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info) {
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
