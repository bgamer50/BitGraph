#pragma once

#define GPU_VERTEX_STEP 0x20

#include "gremlinxx/gremlinxx.h"
#include "step/gpu/GPUTraversalHelper.cuh"

#include <set>

class GPUVertexStep : public TraversalStep {
    private:
        Direction direction;
        std::set<std::string> edge_labels;
        GraphStepType gs_type;
        bool dedup;

    public:
        GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type, bool dedup);

        GPUVertexStep(Direction direction, std::set<std::string> edge_labels, GraphStepType gs_type);

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers);

        using TraversalStep::getInfo;
        virtual std::string getInfo();
};

