#include "step/cpu/CPUSubgraphExtractionStep.h"

#include "structure/CPUGraph.h"

CPUSubgraphExtractionStep::CPUSubgraphExtractionStep(std::string subgraph_name) : TraversalStep(MAP, SUBGRAPH_EXTRACTION_STEP) {
    this->subgraph_name = subgraph_name;
}

void CPUSubgraphExtractionStep::apply(GraphTraversal* traversal, TraverserSet& traversers) {
    CPUGraph* cpu_graph = static_cast<CPUGraph*>(traversal->getGraph());
    CPUGraph* new_cpu_graph = new CPUGraph();

    std::unordered_set<Edge*> edges = boost::any_cast<std::unordered_set<Edge*>>(traversal->getTraversalProperty(SUBGRAPH_PREFIX + subgraph_name));
    std::unordered_map<Vertex*, Vertex*>(old_v_to_new_v);

    for(Edge* e : edges) {
        Vertex* outV = e->outV();
        Vertex* inV = e->inV();
        if(old_v_to_new_v.find(outV) == old_v_to_new_v.end()) {
            old_v_to_new_v[outV] = new_cpu_graph->add_vertex(outV->label());
            for(Property* prop : outV->properties()) {
                auto key = prop->key();
                auto val = prop->value();
                old_v_to_new_v[outV]->property(key, val);
            }
        }
        if(old_v_to_new_v.find(inV) == old_v_to_new_v.end()) {
            old_v_to_new_v[inV] = new_cpu_graph->add_vertex(inV->label()); 
            for(Property* prop : inV->properties()) {
                auto key = prop->key();
                auto val = prop->value();
                old_v_to_new_v[inV]->property(key, val);
            }
        }

        new_cpu_graph->add_edge(old_v_to_new_v[outV], old_v_to_new_v[inV], e->label());
    }
}