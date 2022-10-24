#pragma once

#include "step/TraversalStep.h"
#include "traversal/Traverser.h"
#include "traversal/Scope.h"
#include "traversal/Comparison.h"

#include <functional>

#define GPU_GLOBAL_REDUCTION_STEP 0x26

class GPUGlobalReductionStep: public TraversalStep {
    private:
        gremlinxx::comparison::C comparison_type;
    public:
        GPUGlobalReductionStep(gremlinxx::comparison::C comparison_type)
        : TraversalStep(true, MAP, GPU_GLOBAL_REDUCTION_STEP) {
            this->comparison_type = comparison_type;
        }

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers) {
                gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(traversers.front().get());
                int32_t* gpu_element_traversers = traverser_info.traversers;
        }
};