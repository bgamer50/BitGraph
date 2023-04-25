#pragma once

#include "gremlinxx/gremlinxx.h"

#define CPU_SUBGRAPH_EXTRACTION_STEP 0x15

class CPUSubgraphExtractionStep : public TraversalStep {
    private:
        std::string subgraph_name;

    public:
        CPUSubgraphExtractionStep(std::string subgraph_name);

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers);
};