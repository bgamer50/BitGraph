#ifndef BITGRAPH_STRATEGY_H
#define BITGRAPH_STRATEGY_H

#include "TraversalStrategy.h"
#include "GraphStep.h"
#include "VertexStep.h"

class CPUGraph;

void bitgraph_strategy(CPUGraph* bg, std::vector<TraversalStep*>& steps);

#include "CPUGraph.h"

void bitgraph_strategy(CPUGraph* bg, std::vector<TraversalStep*>& steps) {
    if(steps[0]->uid == GRAPH_STEP) {
		GraphStep* graph_step = (GraphStep*)steps[0];
		if(graph_step->getType() == EDGE) {
			steps[0] = new GraphStep(VERTEX, {});
			steps.emplace(steps.begin() + 1, new VertexStep(OUT, EDGE));
		}
		else if(graph_step->getType() == VERTEX) {
			if(graph_step->get_element_ids().empty()) {
				if(steps[1]->uid == HAS_STEP) {
					HasStep* has_step = static_cast<HasStep*>(steps[1]);
					if(bg->is_indexed(has_step->get_key())) {
						IndexStep* index_list_step = new IndexStep(has_step->get_key(), has_step->get_value());
						delete steps[0];
						steps[0] = index_list_step;
						delete steps[1];
						steps.erase(steps.begin() + 1);
					}
				}
			}
		}
	}
}

#endif