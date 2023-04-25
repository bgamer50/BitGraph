#include "step/hybrid/GPUAddPropertyStep.cuh"

#include "structure/GPUGraph.cuh"
#include "structure/GPUVertex.cuh"
#include "util/gremlin_utils.h"

GPUAddPropertyStep::GPUAddPropertyStep(std::string key, boost::any value)
: TraversalStep(MAP, GPU_ADD_PROPERTY_STEP) {
    this->key = key;
    this->value = value;
}

std::string GPUAddPropertyStep::get_key() { return this->key; }

boost::any GPUAddPropertyStep::get_value() { return this->value; }

std::string GPUAddPropertyStep::getInfo() {
    return "GPUAddPropertyStep{" + this->key + "}";
}

void GPUAddPropertyStep::apply(GraphTraversal* current_traversal, TraverserSet& traversers) {
    GPUGraph* graph = static_cast<GPUGraph*>(current_traversal->getTraversalSource()->getGraph());

    if(this->value.type() == typeid(GraphTraversal*)) {
        GraphTraversal* ap_anonymous_trv = boost::any_cast<GraphTraversal*>(value);
        auto& eid_type = boost::any_cast<Vertex*>(traversers.front().get())->id().type();
        
        for(TraversalStep* step : ap_anonymous_trv->getSteps()) if(step->uid == MIN_STEP) {
            MinStep* min_step = static_cast<MinStep*>(step);
            min_step->set_scope_context(ScopeContext{Scope::local, ADD_PROPERTY_STEP_SIDE_EFFECT_KEY});
        }
        
        for(Traverser& trv : traversers) {
            boost::any v_id = boost::any_cast<Vertex*>(trv.get())->id();
            trv.get_side_effects()[ADD_PROPERTY_STEP_SIDE_EFFECT_KEY] = group_id_from_any(v_id);
        }

        TraverserSet temp_traversers(traversers.begin(), traversers.end());
        GraphTraversal new_trv(current_traversal->getTraversalSource(), ap_anonymous_trv);
        new_trv.setInitialTraversers(temp_traversers);
        new_trv.iterate();
        
        for(Traverser& trv : new_trv.getTraversers()) {
            scope_group_t g_id = boost::any_cast<scope_group_t>(trv.get_side_effects()[ADD_PROPERTY_STEP_SIDE_EFFECT_KEY]);
            boost::any prop_value = trv.get();
            auto v_id = boost::any_cast<uint64_t>(any_from_group_id(g_id, eid_type));
            GPUVertex* gpu_v = static_cast<GPUVertex*>(graph->get_vertex_with_cpu_id(v_id));
            graph->set_property(this->key, gpu_v->gpu_vertex_id, prop_value);
        }
    } 
    else {
        // Store the propety; TODO deal w/ edges
        std::for_each(traversers.begin(), traversers.end(), [&](Traverser& trv){
            GPUVertex* e = static_cast<GPUVertex*>(boost::any_cast<Vertex*>(trv.get()));
            graph->set_property(this->key, e->gpu_vertex_id, this->value);
        });
    }

    // Traversers aren't modified in this step.
}