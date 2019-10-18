#ifndef BITGRAPH_STRATEGY_H
#define BITGRAPH_STRATEGY_H

#include "strategy/TraversalStrategy.h"
#include "step/graph/GraphStep.h"
#include "step/vertex/VertexStep.h"
#include "step/IndexStep.h"
#include "step/HasWithIndexStep.h"

class CPUGraph;

void bitgraph_strategy(CPUGraph* bg, std::vector<TraversalStep*>& steps);

#include "structure/CPUGraph.h"
#include "step/BitGraphStep.h"

void bitgraph_strategy(CPUGraph* bg, std::vector<TraversalStep*>& steps) {
    if(steps[0]->uid == GRAPH_STEP) {
		GraphStep* graph_step = static_cast<GraphStep*>(steps[0]);
		if(graph_step->getType() == EDGE) {
			steps[0] = new GraphStep(VERTEX, {});
			steps.emplace(steps.begin() + 1, new VertexStep(OUT, EDGE));
		}

		graph_step = static_cast<GraphStep*>(steps[0]);
		if(graph_step->getType() == VERTEX) {
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
				else {
					BitGraphStep* bitgraph_step = new BitGraphStep(true, graph_step->getType(), graph_step->get_element_ids());
					steps[0] = bitgraph_step;
				}
			}
			else {
				BitGraphStep* bitgraph_step = new BitGraphStep(true, graph_step->getType(), graph_step->get_element_ids());
				steps[0] = bitgraph_step;
			}
		}
	}

	for(auto it = steps.begin() + 1; it != steps.end(); ++it) {
		TraversalStep* current_step = *it;
		if(current_step->uid == HAS_STEP) {
			HasStep* has_step = static_cast<HasStep*>(current_step);
			if(bg->is_indexed(has_step->get_key())) {
				*it = new HasWithIndexStep(has_step->get_key(), has_step->get_value());
				delete current_step;
			}
		} else if(current_step->uid == GRAPH_STEP) {
			GraphStep* graph_step = static_cast<GraphStep*>(current_step);
			BitGraphStep* bitgraph_step = new BitGraphStep(false, graph_step->getType(), graph_step->get_element_ids());
			*it = bitgraph_step;
		}
	}
}

#endif