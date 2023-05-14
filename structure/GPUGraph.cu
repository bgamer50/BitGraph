#include "structure/GPUGraph.cuh"
#include "structure/GPUVertex.cuh"
#include "structure/GPUEdge.cuh"
#include "structure/BitVertex.h"
#include "structure/BitEdge.h"

GPUGraph::GPUGraph(CPUGraph& cpu_graph): Graph() {
    // Allocate vertex structures
    const size_t num_vertices = cpu_graph.numVertices();
    this->vertex_list.resize(num_vertices);

    // Loop over vertices (note use of access_vertices() for speedup and memory conservation)
    std::vector<Vertex*>& vertices = cpu_graph.access_vertices();
    for(size_t gpu_vertex_id = 0; gpu_vertex_id < num_vertices; ++gpu_vertex_id) {
        BitVertex* v = static_cast<BitVertex*>(vertices[gpu_vertex_id]);
        this->vertex_list[gpu_vertex_id] = new GPUVertex(this, v, gpu_vertex_id);
        this->vertex_id_map[boost::any_cast<uint64_t>(v->id())] = this->vertex_list[gpu_vertex_id];

        for(Property* p : v->properties()) {
            this->set_property(p->key(), gpu_vertex_id, p->value());
        }
    }

    // Pre-set container sizes
    const size_t num_edges = cpu_graph.numEdges();
    this->edge_list.reserve(num_edges);
    this->edge_id_map.reserve(num_edges);

    bitgraph::matrix::sparse_matrix_host<matrix_value_t> M;
    
    M.nnz = num_edges;
    M.num_rows = num_vertices;
    M.num_cols = num_vertices;
    M.values.resize(num_edges, 1);
    M.col_ptr.reserve(num_edges);
    M.row_ptr.reserve(num_vertices + 1);
    M.row_ptr.push_back(0);

    // Loop over vertices
    for(Vertex* v : cpu_graph.vertices()) {
        const uint64_t cpu_out_id = boost::any_cast<uint64_t>(v->id());
        GPUVertex* out = static_cast<GPUVertex*>(this->vertex_id_map[cpu_out_id]);
        
        std::set<size_t> sorted_edges;
        for(Edge* e : v->edges(OUT)) {
            const uint64_t cpu_edge_id = boost::any_cast<uint64_t>(e->id());
            const uint64_t cpu_in_id = boost::any_cast<uint64_t>(e->inV()->id());
            
            GPUVertex* in = static_cast<GPUVertex*>(this->vertex_id_map[cpu_in_id]);
            std::pair<size_t, size_t> eid_gpu = std::make_pair(out->gpu_vertex_id, in->gpu_vertex_id);

            this->edge_list[eid_gpu] = new GPUEdge(cpu_edge_id, e->label(), out, in);
            this->edge_id_map[cpu_edge_id] = this->edge_list[eid_gpu];

            this->edge_label_index[e->label()].push_back(eid_gpu);
            sorted_edges.insert(cpu_in_id);
        }

        // TODO use a sorted map for sorted_edges to get each edge's multiplicity
        M.col_ptr.insert(M.col_ptr.end(), sorted_edges.begin(), sorted_edges.end());
        M.row_ptr.push_back(sorted_edges.size() + M.row_ptr.back());
    }                      

    // Move the adjacency matrix to the GPU and save its pointer
    this->adjacency_matrix = bitgraph::matrix::sparse_convert_host_to_device<matrix_value_t>(M);
}

void GPUGraph::matrix_cache_insert(std::string key, bitgraph::matrix::sparse_matrix_device matrix) {
    if(this->matrix_cache.size() == this->matrix_cache_max_size) {
        auto entry_to_remove = this->matrix_cache.begin();
        size_t highest_usage = this->matrix_cache_usage_tracker[entry_to_remove->first];
        
        for(auto it = this->matrix_cache.begin(); it != this->matrix_cache.end(); ++it) {
            size_t usage = this->matrix_cache_usage_tracker[it->first];
            if(usage >= highest_usage) {
                highest_usage = usage;
                entry_to_remove = it;
            }
        }

        this->matrix_cache_usage_tracker.erase(entry_to_remove->first);
        destroy_sparse_matrix(entry_to_remove->second);
        this->matrix_cache.erase(entry_to_remove);
    }
    this->matrix_cache[key] = std::move(matrix);
    this->matrix_cache_usage_tracker[key] = 0;
}

TypeErasedVector GPUGraph::get_properties(std::string property_key, TypeErasedVector& gpu_vertex_ids, bool strict) {
    if(gpu_vertex_ids.get_dtype() != gremlinxx::comparison::C::UINT64) throw std::runtime_error("Vertices can only be referred to with uint64 values!");
    
    TypeErasedVector retrieved_values;
    std::tie(retrieved_values, std::ignore) = this->property_table.get_property_values(property_key, gpu_vertex_ids, strict=strict);
    return retrieved_values;
}

TypeErasedVector GPUGraph::get_properties(std::string property_key, std::vector<size_t>& gpu_vertex_ids, bool strict) {
    auto type_erased_vertex_ids = bitgraph::memory::make_viewing_vector_from_typed(gpu_vertex_ids);
    return this->get_properties(
        property_key,
        type_erased_vertex_ids,
        strict
    );
}

boost::any GPUGraph::get_property(std::string property_key, size_t gpu_vertex_id, bool strict) {
    std::vector<size_t> gpu_vertex_ids = {gpu_vertex_id};
    TypeErasedVector retrieved_properties = this->get_properties(
        property_key,
        gpu_vertex_ids,
        strict
    );

    if(retrieved_properties.size() == 0) {
        return boost::any();
    }

    auto anys = bitgraph::memory::vector_to_anys(retrieved_properties, &this->string_index);
    return anys.front();
}

void GPUGraph::set_properties(std::string property_key, TypeErasedVector& gpu_vertex_ids, TypeErasedVector& values) {
    if(gpu_vertex_ids.size() != values.size()) throw std::runtime_error("ID array size must match values size");

    // Warn the user if this property is being created implicitly
    if(!this->property_table.has_property_key(property_key)) {
        std::cerr << "Warning: Undefined property " << property_key << ", will create implicitly" << std::endl;
        
        gremlinxx::comparison::C dtype = values.get_dtype();
        size_t initial_max_size = values.size();

        std::stringstream sx;
        sx << "Implicitly creating property " << property_key;
        sx << " with type " << gremlinxx::comparison::C_to_string[dtype];
        sx << " and initial max size " << initial_max_size;
        std::cerr << sx.str() << std::endl;
        this->property_table.declare_property(
            property_key,
            dtype,
            initial_max_size
        );
    }

    this->property_table.set_property_values(
        property_key,
        gpu_vertex_ids,
        values
    );
}

void GPUGraph::set_properties(std::string property_key, std::vector<size_t>& gpu_vertex_ids, std::vector<boost::any>& values) {
    auto type_erased_vertex_ids = bitgraph::memory::make_viewing_vector_from_typed(gpu_vertex_ids);
    
    TypeErasedVector type_erased_values = bitgraph::memory::make_vector_from_anys(values, bitgraph::memory::memory_type::HOST, &this->string_index);
    
    this->set_properties(property_key, type_erased_vertex_ids, type_erased_values);
}

void GPUGraph::set_property(std::string property_key, size_t gpu_vertex_id, boost::any value) {
    std::vector<size_t> gpu_vertex_ids = {gpu_vertex_id};
    std::vector<boost::any> values = {value};
    this->set_properties(property_key, gpu_vertex_ids, values);
}

#include "traversal/GPUGraphTraversalSource.cuh"
GraphTraversalSource* GPUGraph::traversal() { return new GPUGraphTraversalSource(this); }

std::tuple<TypeErasedVector, TypeErasedVector> GPUGraph::query_adjacency(bitgraph::matrix::ADJ adj, Direction dir, TypeErasedVector& input_objects) {
    if(adj != bitgraph::matrix::ADJ::VERTEX_TO_VERTEX) throw std::runtime_error("Only vertex to vertex currently supported for adjacency query!");

    if(dir == Direction::BOTH) {
        cudaError_t err;
        
        cudaStream_t stream_out;
        err = cudaStreamCreateWithFlags(
            &stream_out,
            cudaStreamNonBlocking
        );
        if(err != cudaSuccess) throw std::runtime_error("Failed to create cuda stream");

        cudaStream_t stream_in;
        err = cudaStreamCreateWithFlags(
            &stream_in,
            cudaStreamNonBlocking
        );
        if(err != cudaSuccess) throw std::runtime_error("Failed to create cuda stream");

        bitgraph::matrix::sparse_matrix_device M_OUT = this->get_adjacency_matrix(OUT);
        auto result_out = gpu_query_adjacency_v_to_v(
            M_OUT,
            input_objects,
            stream_out
        );

        bitgraph::matrix::sparse_matrix_device M_IN = this->get_adjacency_matrix(IN);
        auto result_in = gpu_query_adjacency_v_to_v(
            M_IN,
            input_objects,
            stream_in
        );

        cudaStreamSynchronize(stream_out);
        err = cudaStreamDestroy(stream_out);
        if(err != cudaSuccess) throw std::runtime_error("Failed to destroy cuda stream");
        
        cudaStreamSynchronize(stream_in);
        err = cudaStreamDestroy(stream_in);
        if(err != cudaSuccess) throw std::runtime_error("Failed to destroy cuda stream");

        auto& result_out_result_vec = std::get<0>(result_out);
        auto& result_out_origin_vec = std::get<1>(result_out);

        auto& result_in_result_vec = std::get<0>(result_in);
        auto& result_in_origin_vec = std::get<1>(result_in);

        result_out_result_vec.insert(result_out_result_vec.size(), result_in_result_vec);
        result_out_origin_vec.insert(result_out_origin_vec.size(), result_in_origin_vec);


        return result_out;
    }
    else {
        bitgraph::matrix::sparse_matrix_device M = this->get_adjacency_matrix(dir);
        return gpu_query_adjacency_v_to_v(
            M,
            input_objects
        );
    }
}

#include "algorithm/GPUGraphAlgorithm.h"
#include "algorithm/ConnectedComponentsGPUGraphAlgorithm.cuh"
std::unordered_map<std::string, boost::any> GPUGraph::algorithm(GPUGraphAlgorithm* algo) {
    return algo->exec(this);
}