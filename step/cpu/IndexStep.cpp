#include "step/cpu/IndexStep.h"

#include "index/Index.h"
#include "structure/CPUGraph.h"

IndexStep::IndexStep(std::string k, boost::any v)
: TraversalStep(MAP, INDEX_STEP) {
    this->key = k;
    this->value = v;
}

void IndexStep::apply(GraphTraversal* trv, TraverserSet& traversers) {
    Index* idx = dynamic_cast<CPUGraph*>(trv->getGraph())->get_index(this->key);
    std::unordered_set<Element*> elements = idx->get_elements(this->value);
    std::for_each(elements.begin(), elements.end(), [&, this](Element* e) {
        Traverser trv = Traverser((Vertex*)e); // TODO this won't handle edges
        traversers.push_back(trv);
    });
}