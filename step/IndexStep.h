#ifndef INDEX_STEP_H
#define INDEX_STEP_H

#define INDEX_STEP 0x12

#include <boost/any.hpp>
#include "step/TraversalStep.h"

class IndexStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    public:
        IndexStep(std::string key, boost::any value)
        : TraversalStep(MAP, INDEX_STEP) {
            this->key = key;
            this->value = value;
        }

        std::string get_key() {return key;}
        boost::any get_value() {return value;}

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers) {
            Index* idx = dynamic_cast<CPUGraph*>(trv->getGraph())->get_index(this->key);
	        std::unordered_set<Element*> elements = idx->get_elements(this->value);
	        std::for_each(elements.begin(), elements.end(), [&, this](Element* e) {
		        Traverser trv = Traverser((Vertex*)e); // TODO this won't handle edges
		        traversers.push_back(trv);
	        });
        }
};

#endif