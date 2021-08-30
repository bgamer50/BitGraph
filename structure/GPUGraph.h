#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include "structure/matrix/CPUSparseMatrix.h"
#include "structure/matrix/GPUSparseMatrixWrapper.h"

#include "structure/Graph.h"
#include "structure/CPUGraph.h"
#include "structure/Vertex.h"
#include "structure/reference/ReferenceVertex.h"
#include "structure/reference/ReferenceEdge.h"
#include "structure/reference/GPUReferenceVertex.h"
#include "structure/reference/GPUReferenceEdge.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <boost/any.hpp>
#include <boost/functional/hash.hpp>
#include <utility>

// property name -> vertex (gpu id) -> value
typedef std::unordered_map<std::string, std::unordered_map<size_t, boost::any>> property_table_t;

class GPUGraph : public Graph {
    private:
        // cusparse handle, for calling cusparse functions
        cusparseHandle_t cusparse_handle = 0;

        // sparse adjacency matrix
        sparse_matrix_device_t adjacency_matrix;
        
        // property tables
        property_table_t property_table;
        
        // convert CPU ids to sequential ids
        std::vector<Vertex*> vertex_list; // GPU -> CPU
        std::unordered_map<uint64_t, Vertex*> vertex_id_map; // CPU -> GPU

        std::unordered_map<std::pair<int32_t, int32_t>, Edge*, boost::hash<std::pair<int32_t, int32_t>>> edge_list; // GPU -> CPU
        std::unordered_map<uint64_t, Edge*> edge_id_map; // CPU -> GPU

        // index edge labels for subgraph extraction
        std::unordered_map<std::string, std::vector<std::pair<int32_t, int32_t>>> edge_label_index;

    public:
        GPUGraph(CPUGraph& cpu_graph) 
        : Graph() {
            // Allocate vertex structures
            const size_t num_vertices = cpu_graph.numVertices();
            this->vertex_list.resize(num_vertices);

            // Loop over vertices (note use of access_vertices() for speedup and memory conservation)
            std::vector<Vertex*>& vertices = cpu_graph.access_vertices();
            for(int gpu_vertex_id = 0; gpu_vertex_id < num_vertices; ++gpu_vertex_id) {
                BitVertex* v = static_cast<BitVertex*>(vertices[gpu_vertex_id]);
                this->vertex_list[gpu_vertex_id] = new GPUReferenceVertex(v, gpu_vertex_id);
                this->vertex_id_map[boost::any_cast<uint64_t>(v->id())] = this->vertex_list[gpu_vertex_id];

                for(Property* p : v->properties()) {
                    this->property_table[p->key()][gpu_vertex_id] = p->value();
                }
            }

            // Get cusparse handle
            cusparseCreate(&this->cusparse_handle); // TODO may want to get this from somewhere else

            // Pre-set container sizes
            const size_t num_edges = cpu_graph.numEdges();
            this->edge_list.reserve(num_edges);
            this->edge_id_map.reserve(num_edges);

            sparse_matrix_t M;
            
            M.nnz = num_edges;
            M.num_rows = num_vertices;
            M.num_cols = num_vertices;
            M.values.resize(num_edges, 1.0);
            M.col_ptr.reserve(num_edges);
            M.row_ptr.reserve(num_vertices + 1);
            M.row_ptr.push_back(0);

            // Loop over vertices
            for(Vertex* v : cpu_graph.vertices()) {
                const uint64_t cpu_out_id = boost::any_cast<uint64_t>(v->id());
                GPUReferenceVertex* out = static_cast<GPUReferenceVertex*>(this->vertex_id_map[cpu_out_id]);
                
                std::set<int32_t> sorted_edges;
                for(Edge* e : v->edges(OUT)) {
                    const uint64_t cpu_edge_id = boost::any_cast<uint64_t>(e->id());
                    const uint64_t cpu_in_id = boost::any_cast<uint64_t>(e->inV()->id());
                    
                    GPUReferenceVertex* in = static_cast<GPUReferenceVertex*>(this->vertex_id_map[cpu_in_id]);
                    std::pair<int32_t, int32_t> eid_gpu = std::make_pair(out->gpu_vertex_id, in->gpu_vertex_id);

                    this->edge_list[eid_gpu] = new GPUReferenceEdge(cpu_edge_id, e->label(), out, in);
                    this->edge_id_map[cpu_edge_id] = this->edge_list[eid_gpu];

                    this->edge_label_index[e->label()].push_back(eid_gpu);
                    sorted_edges.insert(cpu_in_id);
                }

                M.col_ptr.insert(M.col_ptr.end(), sorted_edges.begin(), sorted_edges.end());
                M.row_ptr.push_back(sorted_edges.size() + M.row_ptr.back());
            }                      

            // Move the adjacency matrix to the GPU and save its pointer
            this->adjacency_matrix = sparse_convert_host_to_device(this->cusparse_handle, M);

        }

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

        sparse_matrix_device_t& access_adjacency_matrix() {
            return this->adjacency_matrix;
        }

        cusparseHandle_t& get_cusparse_handle() {
            return this->cusparse_handle;
        }

        inline boost::any get_property(std::string property_key, size_t gpu_vertex_id) {
            auto it = this->property_table.find(property_key);
            if(it == this->property_table.end()) return boost::any();
            
            auto itt = it->second.find(gpu_vertex_id);
            if(itt == it->second.end()) return boost::any();
            else return boost::any(itt->second);
        }

};

#include "traversal/GPUGraphTraversalSource.h"
GraphTraversalSource* GPUGraph::traversal() { return new GPUGraphTraversalSource(this); }

#endif