#pragma once

#define GPU_ADD_PROPERTY_STEP 0x22

#include "gremlinxx/gremlinxx.h"

// only single cardinality supported
class GPUAddPropertyStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    
    public:
        GPUAddPropertyStep(std::string key, boost::any value);

        std::string get_key();

        boost::any get_value();

        virtual void apply(GraphTraversal* current_traversal, TraverserSet& traversers);

        using TraversalStep::getInfo;
        virtual std::string getInfo();
};

