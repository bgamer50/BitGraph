#ifndef HAS_WITH_INDEX_STEP_H
#define HAS_WITH_INDEX_STEP_H

#define HAS_WITH_INDEX_STEP 0x13

#include <boost/any.hpp>
#include "step/TraversalStep.h"
#include "traversal/Traverser.h"
#include "structure/CPUGraph.h"

class HasWithIndexStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    public:
        HasWithIndexStep(std::string k, boost::any v) : TraversalStep(FILTER, HAS_WITH_INDEX_STEP) {
            this->key = key;
            this->value = value;
        }

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers) {
            Index* idx = static_cast<CPUGraph*>(trv->getGraph())->get_index(this->key);
            std::unordered_set<Element*> elements = idx->get_elements(this->value);
            
            TraverserSet new_traversers;
            for(Traverser trv : traversers) {
                Element* e = static_cast<Element*>(boost::any_cast<Vertex*>(trv.get()));
                bool advance = elements.count(e) > 0;
                if(advance) new_traversers.push_back(trv);
            }

            traversers.swap(new_traversers);
        }
};

#endif