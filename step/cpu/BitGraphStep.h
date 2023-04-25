#pragma once

#include <unordered_map>

#include "gremlinxx/gremlinxx.h"

class BitGraphStep: public GraphStep {
    private:
        bool start;

    public:
        BitGraphStep(bool start, GraphStepType gs_type, std::vector<boost::any> element_ids);

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers);

        using TraversalStep::getInfo;
        virtual std::string	getInfo();
};