#pragma once

#define GPU_UNBIND_STEP 0x2f

#include "gremlinxx/gremlinxx.h"

class GPUUnbindStep : public TraversalStep {
    private:
        bool free_gpu_memory = true;

    public:
        GPUUnbindStep();
        GPUUnbindStep(bool free_gpu_memory);

        virtual void apply(GraphTraversal* parent_traversal, TraverserSet& traversers);

        using TraversalStep::getInfo;
        virtual std::string getInfo();
};
