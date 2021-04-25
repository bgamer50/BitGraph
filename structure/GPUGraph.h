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

// property name -> vertex -> value
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
        std::vector<GPUReferenceVertex*> vertices; // GPU -> CPU
        std::unordered_map<uint64_t, GPUReferenceVertex*> vertex_id_map; // CPU -> GPU

        std::unordered_map<std::pair<int32_t, int32_t>, GPUReferenceEdge*> edges; // GPU -> CPU
        std::unordered_map<uint64_t, GPUReferenceVertex*> edge_id_map; // CPU -> GPU

        // index edge labels for subgraph extraction
        std::unordered_map<std::string, std::vector<std::pair<int32_t, int32_t>>> edge_label_index;

    public:
        GPUGraph(CPUGraph& cpu_graph) 
        : Graph() {
            // Allocate vertex structures
            const size_t num_vertices = cpu_graph.numVertices();
            this->vertices.resize(num_vertices);

            // Loop over vertices (note use of access_vertices() for speedup and memory conservation)
            std::vector<Vertex*>& vertices = cpu_graph.access_vertices();
            for(int gpu_vertex_id = 0; gpu_vertex_id < num_vertices; ++gpu_vertex_id) {
                BitVertex* v = static_cast<BitVertex*>(vertices[gpu_vertex_id]);
                this->vertices[gpu_vertex_id] = new GPUReferenceVertex(v, gpu_vertex_id);
                this->vertex_id_map[boost::any_cast<uint64_t>(v->id())] = this->vertices[gpu_vertex_id];

                for(Property* p : v->properties()) {
                    this->property_table[p->key()][gpu_vertex_id] = p->value();
                }
            }

            // Allocate edge structures
            this->edges.resize(cpu_graph.numEdges());
            cusparseCreate(&this->cusparse_handle); // TODO may want to get this from somewhere else
            sparse_matrix_t M = sparse_make(num_vertices, num_vertices);

            // Loop over edges
            size_t ei = 0;
            for(Edge* e : cpu_graph.edges()) {
                const uint64_t cpu_edge_id = static_cast<uint64_t>(e->id());
                const uint64_t cpu_out_id = boost::any_cast<uint64_t>(e->outV()->id());
                const uint64_t cpu_in_id = boost::any_cast<uint64_t>(e->inV()->id());

                GPUReferenceVertex* out = this->vertices_id_map[cpu_out_id];
                GPUReferenceVertex* in = this->vertices_id_map[cpu_in_id];

                this->edges[ei] = new GPUReferenceEdge(cpu_edge_id, e->label(), out, in);
                this->edge_id_map[cpu_edge_id] = this->edges[ei];

                this->edge_label_index[label].push_back(std::make_pair(out->gpu_vertex_id, in->gpu_vertex_id));

                // Update the adjacency matrix with this edge's info
                sparse_set(M, out->gpu_vertex_id, in->gpu_vertex_id, 1.0);
                
                ++ei;
            }            

            // Move the adjacency matrix to the GPU and save its pointer
            this->adjacency_matrix = sparse_convert_host_to_device(this->cusparse_handle, M);

        }

        virtual GraphTraversalSource* traversal() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual std::vector<Vertex*> vertices() {throw std::runtime_error("Out-of-traversal vertex access unsupported by GPU Graph!");};
        virtual std::vector<Edge*> edges() {throw std::runtime_error("Out-of-traversal edge acess unsupported by GPU Graph!");};
        virtual Vertex* add_vertex(std::string label) {throw std::runtime_error("Vertex addition unsupported by GPU Graph!");};
        virtual Vertex* add_vertex() {throw std::runtime_error("Vertex addition unsupported by GPU Graph!");};
        virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label) {throw std::runtime_error("Edge addition unsupported by GPU Graph!");};

        // GPUGraph-specific accessors
        virtual std::vector<Vertex*>& access_vertices() {
            return this->vertices;
        }

        virtual std::vector<Edge*>& access_edges() {
            return this->edges;
        }

        virtual sparse_matrix_device_t& access_adjacency_matrix() {
            return this->adjacency_matrix;
        }
};

#endif