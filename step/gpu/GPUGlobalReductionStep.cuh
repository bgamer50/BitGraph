#pragma once

#include "gremlinxx/gremlinxx.h"
#include "step/gpu/GPUTraversalHelper.cuh"

#include <functional>

#define GPU_GLOBAL_REDUCTION_STEP 0x26

class GPUGlobalReductionStep: public TraversalStep {
    private:
        gremlinxx::comparison::C comparison_type;
    public:
        GPUGlobalReductionStep(gremlinxx::comparison::C comparison_type);

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers);
};