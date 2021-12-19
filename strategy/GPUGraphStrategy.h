#ifndef GPUGRAPH_STRATEGY_H
#define GPUGRAPH_STRATEGY_H

#include "strategy/TraversalStrategy.h"
#include "strategy/RepeatStepCompletionStrategy.h"

#include "step/graph/GraphStep.h"
#include "step/vertex/VertexStep.h"

#include "step/hybrid/GPUGraphStep.h"
#include "step/gpu/GPUVertexStep.h"
#include "step/gpu/GPUBindStep.h"
#include "step/gpu/GPUUnbindStep.h"
#include "step/hybrid/GPUPropertyStep.h"
#include "step/hybrid/GPUAddPropertyStep.h"

void gpugraph_strategy(std::vector<TraversalStep*>& steps) {
	size_t skip_steps = 0;
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
		
		++skip_steps;
	}

	for(auto it = steps.begin() + skip_steps; it != steps.end(); ++it) {
		TraversalStep* current_step = *it;
		switch(current_step->uid) {
			case GRAPH_STEP: {
				GraphStep* graph_step = static_cast<GraphStep*>(current_step);
				GPUGraphStep* gpugraph_step = new GPUGraphStep(false, graph_step->getType(), graph_step->get_element_ids());
				*it = gpugraph_step;
				break;
			}

			case VERTEX_STEP: {
				VertexStep* vertex_step = static_cast<VertexStep*>(current_step);
				GPUVertexStep* gpu_vertex_step = new GPUVertexStep(vertex_step->get_direction(), vertex_step->get_labels(), vertex_step->get_type());
				if(!gpu_vertex_step->chained) {
					it = steps.insert(it, new GPUBindStep()) + 1;
					it = steps.insert(it + 1, new GPUUnbindStep()) - 1;
					gpu_vertex_step->chained = true;
				}

				*it = gpu_vertex_step;
				break;
			}

			case PROPERTY_STEP: {
				PropertyStep* property_step = static_cast<PropertyStep*>(current_step);
				if(property_step->get_type() == PROPERTY) throw std::runtime_error("Getting properties not supported on GPU, only raw values!");
				
				GPUPropertyStep* gpu_property_step = new GPUPropertyStep(property_step->get_keys());
				*it = gpu_property_step;
				break;
			}

			case ADD_PROPERTY_STEP: {
				AddPropertyStep* add_property_step = static_cast<AddPropertyStep*>(current_step);
				if(add_property_step->get_cardinality() != SINGLE) throw std::runtime_error("Property cardinality other than single not supported on GPU!");
				GPUAddPropertyStep* gpu_add_property_step = new GPUAddPropertyStep(add_property_step->get_key(), add_property_step->get_value());
				*it = gpu_add_property_step;
				break;
			}
			
			case HAS_STEP: {
				HasStep* has_step = static_cast<HasStep*>(current_step);
				has_step->set_acquirer([](GraphTraversalSource* src, Vertex* v, std::string& key){
					GPUGraph* gpu_graph = static_cast<GPUGraph*>(src->getGraph());
					GPUVertex* gpu_v = static_cast<GPUVertex*>(v);
					return gpu_graph->get_property(key, gpu_v->gpu_vertex_id);
				});
				break;
			}

			case REPEAT_STEP: {
				// handle repeat step; if first step is bind and last step is unbind, then move bind before repeat() and unbind after repeat() and place unbind() at beginning of emit() and until()
				// cannot optimize if emit() is used
				RepeatStep* repeat_step = static_cast<RepeatStep*>(current_step);
				if(repeat_step->getEmitTraversal() == nullptr) {
					GraphTraversal* action_traversal = repeat_step->getActionTraversal();
					std::vector<TraversalStep*>& action_steps = action_traversal->getSteps();

					gpugraph_strategy(action_steps);
					if(action_steps.front()->uid == GPU_BIND_STEP && action_steps.back()->uid == GPU_UNBIND_STEP) {
						it = steps.insert(it, new GPUBindStep()) + 1;
						it = steps.insert(it + 1, new GPUUnbindStep()) - 1;

						action_steps.erase(action_steps.begin());
						action_steps.erase(action_steps.begin() + (action_steps.size() - 1));

						GraphTraversal* emit_traversal = repeat_step->getEmitTraversal();
						if(emit_traversal != nullptr) { 
							std::vector<TraversalStep*>& emit_steps = emit_traversal->getSteps();
							emit_steps.insert(emit_steps.begin(), new GPUUnbindStep(false)); 
						}

						GraphTraversal* until_traversal = repeat_step->getUntilTraversal();
						if(until_traversal != nullptr) {
							std::vector<TraversalStep*>& until_steps = until_traversal->getSteps();
							until_steps.insert(until_steps.begin(), new GPUUnbindStep(false));
						}
					}
				}
			}
		}
	}
	

	// Perform GPU Step chaining
	for(auto it = steps.begin(); it != steps.end() - 1; ++it) {
		if((*it)->uid == GPU_UNBIND_STEP && (*(it+1))->uid == GPU_BIND_STEP) {
			it = steps.erase(it, it + 2);
		}
	}

	//for(TraversalStep* step : steps) std::cout << step->uid << std::endl;
	//std::cout << "__________________________________________" << std::endl;
}

#endif