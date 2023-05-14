#include "step/gpu/GPUVertexStep.cuh"

#include "structure/memory/TypeErasure.cuh"
#include "structure/GPUGraph.cuh"
#include "structure/matrix/GPUSparseMatrix.cuh"
#include "step/gpu/GPUTraversalHelper.cuh"
#include <cuda_runtime.h>

GPUVertexStep::GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type, bool dedup)
: TraversalStep(MAP, GPU_VERTEX_STEP) {
    this->direction = direction;
    this->edge_labels = edge_labels;
    this->gs_type = gs_type;
    this->dedup = dedup;
}

GPUVertexStep::GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type) 
: GPUVertexStep(direction, edge_labels, gs_type, false){}

std::string GPUVertexStep::getInfo() { 
    std::string info = "GPUVertexStep{}";

    return info;
}

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

    if(this->gs_type == VERTEX) {
        TypeErasedVector gpu_elements(
            bitgraph::memory::memory_type::DEVICE,
            gremlinxx::comparison::C::UINT64, // implicitly convert to vertex ids
            gpu_element_traversers,
            traverser_info.num_traversers,
            true
        );
        gpu_elements.own();

        auto new_gpu_traversers = gpu_graph->query_adjacency(
            bitgraph::matrix::ADJ::VERTEX_TO_VERTEX,
            this->direction,
            gpu_elements
        );
        
        gpu_elements.clear();

        auto& new_traversers_v = std::get<0>(new_gpu_traversers);
        auto& output_origin_v = std::get<1>(new_gpu_traversers);

        // TODO don't convert back to pointers
        new_traversers_v.disown();
        output_origin_v.disown();
        size_t* new_traversers = static_cast<size_t*>(new_traversers_v.data());
        size_t* output_origin = static_cast<size_t*>(output_origin_v.data());
        size_t num_traversers = new_traversers_v.size();

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
