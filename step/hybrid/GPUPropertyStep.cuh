#pragma once

#define GPU_PROPERTY_STEP 0x21

#include "gremlinxx/gremlinxx.h"
#include <vector>
#include <string>

/**
    Gathers the properties for a particular Vertex.
    Only supports "VALUE" property steps currently.
**/
class GPUPropertyStep: public TraversalStep {
    private:
        std::vector<std::string> keys; //duplicates are allowed, per api

    public:
        GPUPropertyStep(std::vector<std::string> keys);

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers);

        using TraversalStep::getInfo;
        virtual std::string getInfo();
};
