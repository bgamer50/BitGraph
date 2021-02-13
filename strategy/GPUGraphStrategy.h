#ifndef GPUGRAPH_STRATEGY_H
#define GPUGRAPH_STRATEGY_H

#include "strategy/TraversalStrategy.h"
#include "step/graph/GraphStep.h"
#include "step/vertex/VertexStep.h"
#include "step/IndexStep.h"
#include "step/HasWithIndexStep.h"

class GPUGraph;

void gpugraph_strategy(GPUGraph* graph, std::vector<TraversalStep*>& steps);

#include "structure/GPUGraph.h"

void bitgraph_strategy(GPUGraph* graph, std::vector<TraversalStep*>& steps) {
    if(steps[0]->uid == GRAPH_STEP) {
		GraphStep* graph_step = static_cast<GraphStep*>(steps[0]);
		if(graph_step->getType() == EDGE) {
			steps[0] = new GraphStep(VERTEX, {});
			steps.emplace(steps.begin() + 1, new VertexStep(OUT, EDGE));
		}

		// at this point, we know it must be a GraphStep{Vertex}.
		graph_step = static_cast<GraphStep*>(steps[0]);
        GPUGraphStep* gpugraph_step = new GPUGraphStep(true, graph_step->getType(), graph_step->get_element_ids());
		steps[0] = gpugraph_step;
	}

	for(auto it = steps.begin() + 1; it != steps.end(); ++it) {
		TraversalStep* current_step = *it;
        if(current_step->uid == GRAPH_STEP) {
			GraphStep* graph_step = static_cast<GraphStep*>(current_step);
			GPUGraphStep* gpugraph_step = new GPUGraphStep(false, graph_step->getType(), graph_step->get_element_ids());
			*it = gpugraph_step;
		}
	}
}

#endif