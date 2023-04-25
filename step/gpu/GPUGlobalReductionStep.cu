#include "step/gpu/GPUGlobalReductionStep.cuh"

GPUGlobalReductionStep::GPUGlobalReductionStep(gremlinxx::comparison::C comparison_type)
: TraversalStep(true, MAP, GPU_GLOBAL_REDUCTION_STEP) {
    this->comparison_type = comparison_type;
}

void GPUGlobalReductionStep::apply(GraphTraversal* traversal, TraverserSet& traversers) {
    gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(traversers.front().get());

    // these should be uint64 right?
    int32_t* gpu_element_traversers = static_cast<int32_t*>(traverser_info.traversers);
}