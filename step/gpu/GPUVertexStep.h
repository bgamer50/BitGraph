#ifndef GPU_VERTEX_STEP_H
#define GPU_VERTEX_STEP_H

#define GPU_VERTEX_STEP 0x20

#include "step/TraversalStep.h"
#include "step/vertex/VertexStep.h"

#include <set>

class GPUVertexStep : public TraversalStep {
    private:
        Direction direction;
        std::set<std::string> edge_labels;
        GraphStepType gs_type;
        bool dedup;

    public:
        GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type, bool dedup)
        : TraversalStep(MAP, GPU_VERTEX_STEP) {
            this->direction = direction;
            this->edge_labels = edge_labels;
            this->gs_type = gs_type;
            this->dedup = dedup;
        }

        GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type) 
        : GPUVertexStep(direction, edge_labels, gs_type, false){}

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers);
};

#include "structure/GPUGraph.h"
#include "structure/matrix/GPUSparseMatrix.h"
#include "step/gpu/GPUTraversalHelper.h"
#include "traversal/Comparison.h"
#include <cuda_runtime.h>

void GPUVertexStep::apply(GraphTraversal* traversal, TraverserSet& traversers) {
    gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(traversers.front().get());
    size_t* gpu_element_traversers = static_cast<size_t*>(traverser_info.traversers);

    // short-circuit if there are no traversers
    if(traverser_info.num_traversers == 0) {
        traversers.clear();
        return;
    }

    if(traverser_info.traverser_dtype != gremlinxx::comparison::C::VERTEX) {
        std::stringstream ss;
        ss << "Encountered an illegal traverser type for VertexStep: ";
        ss << gremlinxx::comparison::C_to_string[traverser_info.traverser_dtype];
        throw std::runtime_error(ss.str());
    }

    GPUGraph* gpu_graph = static_cast<GPUGraph*>(traversal->getGraph());

    // Manipulate the graph's sparse matrix directly
    bitgraph::matrix::sparse_matrix_device& adjacency_matrix = gpu_graph->access_adjacency_matrix();

    if(this->gs_type == VERTEX) {
        // (vertex id) -> (originating traverser)
        std::tuple<size_t*, size_t*, size_t> new_gpu_traversers;
        // TODO for the both matrix you need to call the adjacency query twice on separate streams
        bitgraph::matrix::sparse_matrix_device adj_mat_direction = gpu_graph->get_adjacency_matrix(this->direction);
        new_gpu_traversers = gpu_query_adjacency_v_to_v(adj_mat_direction, gpu_element_traversers, traverser_info.num_traversers);
        cudaFree(gpu_element_traversers);
        cudaDeviceSynchronize();
        cudaCheckErrors("free gpu element traversers");

        auto& new_traversers = std::get<0>(new_gpu_traversers);
        auto& output_origin = std::get<1>(new_gpu_traversers);
        auto& num_traversers = std::get<2>(new_gpu_traversers);
        if(this->dedup) {
            size_t* V_ptr; // allocated by pick_unique
            size_t n_unique;
            std::tie(V_ptr, n_unique) = pick_unique(
                new_traversers,
                num_traversers
            );

            traverser_info.num_traversers = n_unique;
            traverser_info.traversers = new_traversers; // was modified by pick_unique
            traverser_info.paths.push_back(std::make_pair(output_origin, num_traversers));
            traverser_info.paths.push_back(std::make_pair(V_ptr, traverser_info.num_traversers));
        }
        else {
            traverser_info.num_traversers = num_traversers;
            traverser_info.traversers = new_traversers;
            traverser_info.paths.push_back(std::make_pair(output_origin, traverser_info.num_traversers));
        }

        traversers.front().replace_data(traverser_info);
    } else {
        // (outV id, inV id) -> (originating traverser)
        //std::pair<std::pair<int32_t*, int32_t*>, int32_t*> new_gpu_traversers;
        //gpu_query_adjacency_v_to_e(adjacency_matrix, gpu_element_traversers);
        throw std::runtime_error("Vertex to edge query currently unsupported");
    }
}

#endif