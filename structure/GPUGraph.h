#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include "structure/matrix/CPUSparseMatrix.h"
#include "structure/matrix/GPUSparseMatrixWrapper.h"

#include "structure/Graph.h"
#include "structure/CPUGraph.h"
#include "structure/Vertex.h"
#include "structure/reference/ReferenceVertex.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <boost/any.hpp>

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
        std::vector<uint64_t> vertex_ids; // GPU -> CPU
        std::unordered_map<uint64_t, size_t> vertex_id_map; // CPU -> GPU

        std::vector<uint64_t> edge_ids; // GPU -> CPU
        std::unordered_map<uint64_t, size_t> edge_id_map; // CPU -> GPU

        // vertex labels
        std::vector<std::string> vertex_labels;

        // edge labels
        std::vector<std::string> edge_labels;

        // index edge labels for subgraph extraction
        std::unordered_map<std::string, std::vector<uint64_t>> edge_label_index;

    public:
        GPUGraph(CPUGraph& cpu_graph) 
        : Graph() {
            // Allocate vertex structures
            this->vertex_ids.resize(cpu_graph.numVertices());
            this->vertex_labels.resize(cpu_graph.numVertices());

            // Loop over vertices (note use of access_vertices() for speedup and memory conservation)
            std::vector<Vertex*>& vertices = cpu_graph.access_vertices();
            for(int k = 0; k < cpu_graph.numVertices(); ++k) {
                Vertex* v = vertices[k];
                uint64_t vid = boost::any_cast<uint64_t>(v->id());
                this->vertex_labels[k] = v->label();

                this->vertex_ids[k] = vid;
                this->vertex_id_map[vid] = k;

                for(Property* p : v->properties()) {
                    this->property_table[p->key()][k] = p->value();
                }
            }

            // Allocate edge structures
            this->edge_labels.resize(cpu_graph.numEdges());
            cusparseCreate(&this->cusparse_handle); // TODO may want to get this from somewhere else
            sparse_matrix_t M = sparse_make(this->vertex_ids.size(), this->vertex_ids.size());

            // Loop over edges
            size_t ei = 0;
            for(Edge* e : cpu_graph.edges()) {
                this->edge_ids[ei] = e->id();
                this->edge_id_map[e->id()] = ei;

                std::string label = e->label();
                this->edge_labels[ei] = label;
                this->edge_label_index[label].push_back(ei);

                int32_t out = this->vertex_id_map[boost::any_cast<uint64_t>(e->outV()->id())];
                int32_t in = this->vertex_id_map[boost::any_cast<uint64_t>(e->inV()->id())];
                sparse_set(M, out, in, 1.0);
                
                ++ei;
            }            

            this->adjacency_matrix = sparse_convert_host_to_device(this->cusparse_handle, M);

        }

        virtual GraphTraversalSource* traversal() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual std::vector<Vertex*> vertices() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual std::vector<Edge*> edges() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual Vertex* add_vertex(std::string label) {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual Vertex* add_vertex() {throw std::runtime_error("Unsupported by GPU Graph!");};
        virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label) {throw std::runtime_error("Unsupported by GPU Graph!");};

        // GPUGraph-specific accessors
        // FIXME this is a terrible memory leak that needs to be corrected using std::shared_ptr
        virtual std::vector<ReferenceVertex*> view_vertices() {
            size_t num_vertices = this->vertex_ids.size();

            std::vector<ReferenceVertex*> vertex_view(num_vertices);
            for(size_t gpuid = 0; gpuid < num_vertices; ++gpuid) vertex_view[gpuid] = new ReferenceVertex(this->vertex_ids[gpuid], this->vertex_labels[gpuid]);

            return vertex_view;
        }

};

#endif