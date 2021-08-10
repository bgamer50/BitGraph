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
    GPUGraph* gpu_graph = static_cast<GPUGraph*>(traversal->getGraph());

    // Manipulate the graph's sparse matrix directly
    sparse_matrix_device_t& adjacency_matrix = gpu_graph->access_adjacency_matrix();
    if(this->direction == IN) {
        adjacency_matrix = transpose_csr_matrix(gpu_graph->get_cusparse_handle(), adjacency_matrix);
    }

    int32_t* gpu_element_traversers = to_gpu(traversers);

    if(this->gs_type == VERTEX) {
        // (vertex id) -> (originating traverser)
        std::tuple<int32_t*, int32_t*, int> new_gpu_traversers = gpu_query_adjacency_v_to_v(adjacency_matrix, gpu_element_traversers, traversers.size());
        
        int32_t* traversed_vertices_device = std::get<0>(new_gpu_traversers);
        int32_t* originating_traversers_device = std::get<1>(new_gpu_traversers);
        int num_new_traversers = std::get<2>(new_gpu_traversers);

        std::vector<int32_t> new_traversed_vertices(num_new_traversers);
        std::vector<int32_t> originating_traversers(num_new_traversers);
        cudaMemcpy(new_traversed_vertices.data(), traversed_vertices_device, sizeof(int32_t) * num_new_traversers, cudaMemcpyDeviceToHost);
        cudaMemcpy(originating_traversers.data(), originating_traversers_device, sizeof(int32_t) * num_new_traversers, cudaMemcpyDeviceToHost);

        // Free all this step's remaining device objects
        cudaFree(traversed_vertices_device);
        cudaFree(originating_traversers_device);
        
        TraverserSet new_traversers(num_new_traversers);
        for(int k = 0; k < num_new_traversers; ++k) {
            Vertex* v = static_cast<Vertex*>(gpu_graph->access_vertices()[new_traversed_vertices[k]]);
            Traverser& originating_traverser = traversers[originating_traversers[k]];
            
            new_traversers[k].replace_data(v);
            auto& se = originating_traverser.get_side_effects();
            new_traversers[k].get_side_effects().insert(se.begin(), se.end());
        }
        traversers.swap(new_traversers);
    } else {
        // (outV id, inV id) -> (originating traverser)
        std::pair<std::pair<int32_t*, int32_t*>, int32_t*> new_gpu_traversers;
        gpu_query_adjacency_v_to_e(adjacency_matrix, gpu_element_traversers);
    }

    // free the used GPU memory (we assume no chaining here)
    cudaFree(gpu_element_traversers);
}

#endif