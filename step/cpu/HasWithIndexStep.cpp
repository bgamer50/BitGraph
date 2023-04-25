#include "step/cpu/HasWithIndexStep.h"

#include "structure/CPUGraph.h"
#include "index/Index.h"

HasWithIndexStep::HasWithIndexStep(std::string key, boost::any value) : TraversalStep(FILTER, HAS_WITH_INDEX_STEP) {
    this->key = key;
    this->value = value;
}

void HasWithIndexStep::apply(GraphTraversal* trv, TraverserSet& traversers) {
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