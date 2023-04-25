#pragma once

#include <vector>
#include <boost/any.hpp>
#include "gremlinxx/gremlinxx.h"

class GPUGraphStep: public GraphStep {
    private:
        bool start;

    public:
        GPUGraphStep(bool start, GraphStepType gs_type, std::vector<boost::any> element_ids);

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers);
};

