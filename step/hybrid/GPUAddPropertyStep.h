#ifndef GPU_ADD_PROPERTY_STEP_H
#define GPU_ADD_PROPERTY_STEP_H

#define GPU_ADD_PROPERTY_STEP 0x22

#include "structure/GPUGraph.h"
#include "step/property/AddPropertyStep.h"
#include "util/gremlin_utils.hpp"

// only single cardinality supported
class GPUAddPropertyStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    
    public:
        GPUAddPropertyStep(std::string key, boost::any value)
        : TraversalStep(MAP, GPU_ADD_PROPERTY_STEP) {
            this->key = key;
            this->value = value;
        }

        std::string get_key() { return this->key; }

        boost::any get_value() { return this->value; }

        virtual void apply(GraphTraversal* current_traversal, TraverserSet& traversers) {
            GPUGraph* graph = static_cast<GPUGraph*>(current_traversal->getTraversalSource()->getGraph());

			if(this->value.type() == typeid(GraphTraversal*)) {
				GraphTraversal* ap_anonymous_trv = boost::any_cast<GraphTraversal*>(value);
				
                // for each current traverser in the traversal...
				for(Traverser& trv : traversers) {
                    //Element* e = get_element(trv->get());
					GPUVertex* e = static_cast<GPUVertex*>(boost::any_cast<Vertex*>(trv.get())); // allow only traversal of vertices for now
					GraphTraversal new_trv(current_traversal->getTraversalSource(), ap_anonymous_trv);
					
					// Execute traversal
					new_trv.setInitialTraversers({Traverser(trv)});
					boost::any prop_value = new_trv.first();

					// Store the property; TODO deal w/ edges
                    graph->set_property(this->key, e->gpu_vertex_id, prop_value);
				}
			} 
			else {
				// Store the propety; TODO deal w/ edges
				std::for_each(traversers.begin(), traversers.end(), [&](Traverser& trv){
					GPUVertex* e = static_cast<GPUVertex*>(boost::any_cast<Vertex*>(trv.get()));
                    graph->set_property(this->key, e->gpu_vertex_id, this->value);
				});
				//std::cout << "property stored!\n";
			}

			// Traversers aren't modified in this step.
		}

        std::string getInfo() {
            return "GPUAddPropertyStep{" + key + "}";
        }
};

#endif