#include "step/gpu/GPUUnbindStep.cuh"
#include "step/gpu/GPUTraversalHelper.cuh"

GPUUnbindStep::GPUUnbindStep()
: TraversalStep(MAP, GPU_UNBIND_STEP) {}

GPUUnbindStep::GPUUnbindStep(bool free_gpu_memory)
: TraversalStep(MAP, GPU_UNBIND_STEP) {
    this->free_gpu_memory = free_gpu_memory;
}

void GPUUnbindStep::apply(GraphTraversal* parent_traversal, TraverserSet& traversers) {
    TraverserSet new_traversers;
    for(Traverser& trv : traversers) {
        if(trv.get().type() == typeid(gpu_traverser_info_t)) {
            gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(trv.get());
            
            // short-circuit if there are no traversers
            if(traverser_info.num_traversers == 0) {
                continue;
            }

            C_RETRIEVE_NEW_TRAVERSERS(parent_traversal, new_traversers, traverser_info);
        }
        else {
            new_traversers.push_back(trv);
        }
    }
    
    traversers.swap(new_traversers);
    //std::cout << "leave unbind step " << std::endl;
}

std::string GPUUnbindStep::getInfo() {
    return "GPUUnbindStep{}";
}