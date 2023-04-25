#pragma once

#define GPU_BIND_STEP 0x2e

#include "gremlinxx/gremlinxx.h"

#include <numeric>

class GPUBindStep : public TraversalStep {
    private:
        gremlinxx::comparison::C dtype;
    public:
        GPUBindStep(gremlinxx::comparison::C dtype);

        GPUBindStep();

        virtual void apply(GraphTraversal* parent_traversal, TraverserSet& traversers);

        using TraversalStep::getInfo;
        virtual std::string getInfo();
};