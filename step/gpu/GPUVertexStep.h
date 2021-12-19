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

    public:
        // whether this step has gone through the chaining process
        bool chained = false;

        GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type)
        : TraversalStep(MAP, GPU_VERTEX_STEP) {
            this->direction = direction;
            this->edge_labels = edge_labels;
            this->gs_type = gs_type;
        }

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers);
};

#include "structure/GPUGraph.h"
#include "structure/matrix/GPUSparseMatrixWrapper.h"
#include "step/gpu/GPUTraversalHelper.h"
#include <cuda_runtime.h>

void GPUVertexStep::apply(GraphTraversal* traversal, TraverserSet& traversers) {
    gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(traversers.front().get());
    int32_t* gpu_element_traversers = traverser_info.traversers;

    // short-circuit if there are no traversers
    if(traverser_info.num_traversers == 0) {
        traversers.clear();
        return;
    }

    GPUGraph* gpu_graph = static_cast<GPUGraph*>(traversal->getGraph());

    // Manipulate the graph's sparse matrix directly
    sparse_matrix_device_t& adjacency_matrix = gpu_graph->access_adjacency_matrix();

    if(this->gs_type == VERTEX) {
        // (vertex id) -> (originating traverser)
        std::tuple<int32_t*, int32_t*, int> new_gpu_traversers;
        sparse_matrix_device_t adj_mat_direction = gpu_graph->get_adjacency_matrix(this->direction);
        new_gpu_traversers = gpu_query_adjacency_v_to_v(adj_mat_direction, gpu_element_traversers, traverser_info.num_traversers);
        cudaFree(gpu_element_traversers);
        cudaDeviceSynchronize();
        cudaCheckErrors("free gpu element traversers");
        
        traverser_info.num_traversers = std::get<2>(new_gpu_traversers);
        traverser_info.traversers = std::get<0>(new_gpu_traversers);
        traverser_info.paths.push_back(std::make_pair(std::get<1>(new_gpu_traversers), traverser_info.num_traversers));

        traversers.front().replace_data(traverser_info);
    } else {
        // (outV id, inV id) -> (originating traverser)
        std::pair<std::pair<int32_t*, int32_t*>, int32_t*> new_gpu_traversers;
        gpu_query_adjacency_v_to_e(adjacency_matrix, gpu_element_traversers);
    }
}

#endif