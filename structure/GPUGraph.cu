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
    for(int gpu_vertex_id = 0; gpu_vertex_id < num_vertices; ++gpu_vertex_id) {
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

        std::cout << "returned to query adjacency" << std::endl;

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

        std::cout << "reached return statement in query adjacency" << std::endl;
        //std::cout << "this is to trick the compiler " << result_in_result_vec.size() << " " << result_in_origin_vec.size() << std::endl;

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