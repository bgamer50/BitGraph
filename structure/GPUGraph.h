#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include "structure/matrix/CPUSparseMatrix.h"
#include "structure/matrix/GPUSparseMatrixWrapper.h"

#include "structure/Graph.h"
#include "structure/CPUGraph.h"
#include "structure/Vertex.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <boost/any.hpp>

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
        std::vector<uint64_t> vertex_ids;
        std::unordered_map<uint64_t, size_t> vertex_id_map;

        // index edge labels for subgraph extraction
        std::vector<std::string, std::vector<uint64_t>> edge_label_index;

    public:
        GPUGraph(CPUGraph& cpu_graph) 
        : Graph() {
            this->vertex_ids.resize(cpu_graph.numVertices());
            cusparseCreate(&cusparse_handle);

            std::vector<Vertex*>& vertices = cpu_graph.access_vertices();
            for(int k = 0; k < cpu_graph.numVertices(); ++k) {
                Vertex* v = vertices[k];
                uint64_t vid = boost::any_cast<uint64_t>(v->id());

                this->vertex_ids[k] = vid;
                this->vertex_id_map[vid] = k;
            }

            sparse_matrix_t M = sparse_make(this->vertex_ids.size(), this->vertex_ids.size());
            for(Edge* e : cpu_graph.edges()) {
                int32_t out = this->vertex_id_map[boost::any_cast<uint64_t>(e->outV()->id())];
                int32_t in = this->vertex_id_map[boost::any_cast<uint64_t>(e->inV()->id())];
                sparse_set(M, out, in, 1.0);
            }            

            this->adjacency_matrix = sparse_convert_host_to_device(cusparseHandle, M);

            // TODO - property tables
        }

        virtual GraphTraversalSource* traversal() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual std::vector<Vertex*> vertices() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual std::vector<Edge*> edges() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual Vertex* add_vertex(std::string label) {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual Vertex* add_vertex() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label) {throw std::runtime_error("Unsupported by GPU Graph!");};

};

#endif