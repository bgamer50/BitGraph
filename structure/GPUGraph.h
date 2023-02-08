#pragma once

#include "structure/matrix/CPUSparseMatrix.h"
#include "structure/matrix/GPUSparseMatrix.h"
#include "structure/memory/GPUPropertyTable.h"
class GPUGraphAlgorithm;

#include "structure/Graph.h"
#include "structure/CPUGraph.h"
#include "structure/Vertex.h"

#include <cuda_runtime.h>

#include <boost/any.hpp>
#include <boost/functional/hash.hpp>
#include <utility>

#include <unordered_map>
#include <vector>

#define IN_ADJACENCY_MATRIX "IN_ADJACENCY_MATRIX"
#define BOTH_ADJACENCY_MATRIX "BOTH_ADJACENCY_MATRIX"

class GPUVertex;
class GPUEdge;

// The value type for this graph's sparse matrices
// Holds the multiplicity of each edge
// Can be substituted for some other values by
// shallow-copying the matrix and replacing values
// (and setting the right dtype, of course).
using matrix_value_t = uint32_t;

class GPUGraph : public Graph {
    private:
        // sparse adjacency matrix
        bitgraph::matrix::sparse_matrix_device adjacency_matrix;
        
        // property tables
        bitgraph::memory::GPUPropertyTable property_table;
        
        // convert CPU ids to sequential ids
        std::vector<Vertex*> vertex_list; // GPU -> CPU
        std::unordered_map<uint64_t, Vertex*> vertex_id_map; // CPU -> GPU

        std::unordered_map<std::pair<size_t, size_t>, Edge*, boost::hash<std::pair<size_t, size_t>>> edge_list; // GPU -> CPU
        std::unordered_map<uint64_t, Edge*> edge_id_map; // CPU -> GPU

        // index edge labels for subgraph extraction
        std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> edge_label_index;

        // whether to cache various matrices (i.e. both-adjacency matrix, in-adjacency matrix)
        bool matrix_caching_enabled = true;
        size_t matrix_cache_max_size = 3;

        // matrix cache
        std::unordered_map<std::string, bitgraph::matrix::sparse_matrix_device> matrix_cache;
        std::unordered_map<std::string, size_t> matrix_cache_usage_tracker;

        std::optional<bitgraph::matrix::sparse_matrix_device> matrix_cache_find(std::string key) { 
            for(auto it = matrix_cache_usage_tracker.begin(); it != matrix_cache_usage_tracker.end(); ++it) ++(it->second);
            
            auto f = matrix_cache.find(key);
            if(f == matrix_cache.end()) return {};
            
            matrix_cache_usage_tracker[key] = 0;
            return f->second;            
        }

    public:
        GPUGraph(CPUGraph& cpu_graph);

        virtual GraphTraversalSource* traversal();


        virtual std::vector<Vertex*> vertices() {throw std::runtime_error("Out-of-traversal vertex access unsupported by GPU Graph!");};
        virtual std::vector<Edge*> edges() {throw std::runtime_error("Out-of-traversal edge acess unsupported by GPU Graph!");};
        virtual Vertex* add_vertex(std::string label) {throw std::runtime_error("Vertex addition unsupported by GPU Graph!");};
        virtual Vertex* add_vertex() {throw std::runtime_error("Vertex addition unsupported by GPU Graph!");};
        virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label) {throw std::runtime_error("Edge addition unsupported by GPU Graph!");};

        // GPUGraph-specific accessors
        std::vector<Vertex*>& access_vertices() {
            return this->vertex_list;
        }

        Vertex* get_vertex_with_cpu_id(uint64_t vid) {
            return this->vertex_id_map[vid];
        }

        std::vector<Edge*>& access_edges() {
            throw std::runtime_error("Cannot currently access edges!");
        }

        Edge* get_edge_with_cpu_id(uint64_t eid) {
            return this->edge_id_map[eid];
        }

        void matrix_cache_insert(std::string key, bitgraph::matrix::sparse_matrix_device value) {
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
            this->matrix_cache[key] = value;
            this->matrix_cache_usage_tracker[key] = 0;
        }

        bitgraph::matrix::sparse_matrix_device get_in_adjacency_matrix() {
            auto in_adjacency_matrix_opt = this->matrix_cache_find(IN_ADJACENCY_MATRIX);
            if(in_adjacency_matrix_opt) return in_adjacency_matrix_opt.value();

            bitgraph::matrix::sparse_matrix_device in_adjacency_matrix = transpose_csr_matrix(this->adjacency_matrix);
            if(this->matrix_caching_enabled) this->matrix_cache_insert(IN_ADJACENCY_MATRIX, in_adjacency_matrix);
            return in_adjacency_matrix;
        }

        bitgraph::matrix::sparse_matrix_device get_adjacency_matrix(Direction dir) {
            switch(dir) {
                case OUT:
                    return this->adjacency_matrix;
                case IN: 
                    return this->get_in_adjacency_matrix();
                case BOTH: 
                    throw std::runtime_error("Can only get adjacency matrix for OUT or IN direction");
            }

            throw std::runtime_error("Illegal direction provided to get_adjacency_matrix");
        }

        bitgraph::memory::GPUPropertyTable& access_property_table() {
            return this->property_table;
        }

        bitgraph::matrix::sparse_matrix_device& access_adjacency_matrix() {
            return this->adjacency_matrix;
        }

        bool is_matrix_caching_enabled() {
            return this->matrix_caching_enabled;
        }

        void set_matrix_caching(bool should_cache) {
            if(!should_cache) {
                this->matrix_cache.clear();
                this->matrix_cache_usage_tracker.clear();
            }
            this->matrix_caching_enabled = should_cache;
        }

        inline size_t num_vertices() {
            return this->vertex_list.size();
        }

        inline std::vector<boost::any> get_properties(std::string property_key, std::vector<size_t> gpu_vertex_ids) {
            return this->property_table.get_property_values(gpu_vertex_ids, property_key);
        }

        inline boost::any get_property(std::string property_key, size_t gpu_vertex_id) {
            return this->property_table.get_property_value(gpu_vertex_id, property_key);
        }

        inline void set_property(std::string property_key, size_t gpu_vertex_id, boost::any value) {
            if(gpu_vertex_id >= this->vertex_list.size()) {
                throw std::runtime_error("Invalid Vertex ID!");
            }

            // Warn the user if this property is being created implicitly
            if(!this->property_table.has_property_key(property_key)) {
                std::cerr << "Warning: Undefined property " << property_key << ", will create implicitly" << std::endl;
                
                gremlinxx::comparison::C dtype = gremlinxx::comparison::from_any(value);
                size_t initial_max_size = this->num_vertices();

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

            this->property_table.set_property_value(gpu_vertex_id, property_key, value);
        }

        inline void declare_property(std::string property_name, gremlinxx::comparison::C property_dtype, size_t initial_max_size) {
            this->property_table.declare_property(property_name, property_dtype, initial_max_size);
        }

       std::unordered_map<std::string, boost::any> algorithm(GPUGraphAlgorithm* algo);
};

#include "structure/GPUVertex.h"
#include "structure/GPUEdge.h"

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

#include "traversal/GPUGraphTraversalSource.h"
GraphTraversalSource* GPUGraph::traversal() { return new GPUGraphTraversalSource(this); }

#include "algorithm/GPUGraphAlgorithm.h"
#include "algorithm/ConnectedComponentsGPUGraphAlgorithm.h"
std::unordered_map<std::string, boost::any> GPUGraph::algorithm(GPUGraphAlgorithm* algo) {
    return algo->exec(this);
}
