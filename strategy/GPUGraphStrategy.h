#ifndef GPUGRAPH_STRATEGY_H
#define GPUGRAPH_STRATEGY_H

#include "strategy/TraversalStrategy.h"
#include "step/graph/GraphStep.h"
#include "step/vertex/VertexStep.h"

#include "step/hybrid/GPUGraphStep.h"
#include "step/gpu/GPUVertexStep.h"
#include "step/hybrid/GPUPropertyStep.h"

void gpugraph_strategy(std::vector<TraversalStep*>& steps) {
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
		} else if(current_step->uid == VERTEX_STEP) {
			VertexStep* vertex_step = static_cast<VertexStep*>(current_step);
			GPUVertexStep* gpu_vertex_step = new GPUVertexStep(vertex_step->get_direction(), vertex_step->get_labels(), vertex_step->get_type());
			*it = gpu_vertex_step;
		} else if(current_step->uid == PROPERTY_STEP) {
			PropertyStep* property_step = static_cast<PropertyStep*>(current_step);
			if(property_step->get_type() == PROPERTY) throw std::runtime_error("Getting properties not supported on GPU, only raw values!");
			
			GPUPropertyStep* gpu_property_step = new GPUPropertyStep(property_step->get_keys());
			*it = gpu_property_step;
		}
	}
}

#endif