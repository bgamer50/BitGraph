#include "step/gpu/GPUBindStep.cuh"

#include "step/gpu/GPUTraversalHelper.cuh"
#include "util/cuda_utils.cuh"

GPUBindStep::GPUBindStep(gremlinxx::comparison::C dtype)
: TraversalStep(MAP, GPU_BIND_STEP) {
    this->dtype = dtype;
}

GPUBindStep::GPUBindStep()
: TraversalStep(MAP, GPU_BIND_STEP) {
    this->dtype = gremlinxx::comparison::C::INT32;
}

std::string GPUBindStep::getInfo() {
    std::stringstream ss;
    ss << "GPUBindStep{" << gremlinxx::comparison::C_to_string[this->dtype] << "}";
    return ss.str();
}

void GPUBindStep::apply(GraphTraversal* parent_traversal, TraverserSet& traversers) {
    gpu_traverser_info_t traverser_info;
    traverser_info.traversers = C_TO_GPU(this->dtype, traversers);
    traverser_info.traverser_dtype = this->dtype;

    size_t* originating_traversers;
    cudaMalloc((void**) &originating_traversers, sizeof(size_t) * traversers.size());
    cudaDeviceSynchronize();
    cudaCheckErrors("allocate originating traversers");

    std::vector<size_t> h_originating_traversers(traversers.size());
    std::iota(h_originating_traversers.begin(), h_originating_traversers.end(), 0);
    cudaMemcpy(originating_traversers, h_originating_traversers.data(), sizeof(size_t) * traversers.size(), cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaCheckErrors("copy originating traversers");
    
    traverser_info.paths.push_back(std::make_pair(originating_traversers, traversers.size()));

    traverser_info.num_traversers = traversers.size();
    traverser_info.original_traversers.swap(traversers);

    traversers.push_back(Traverser(traverser_info));
}