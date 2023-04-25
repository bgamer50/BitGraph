#include "step/cpu/BitGraphStep.h"

#include "structure/CPUGraph.h"

BitGraphStep::BitGraphStep(bool start, GraphStepType gs_type, std::vector<boost::any> element_ids) 
: GraphStep(gs_type, element_ids) {
    this->start = start;
}

std::string	BitGraphStep::getInfo() {
    std::string info = "BitGraphStep(";
    info = info + (this->element_ids.size() > 0 ? "{...}" : "{}");
    return info + ")";
}

void BitGraphStep::apply(GraphTraversal* trv, TraverserSet& traversers) {
    if(start) traversers.push_back(Traverser(boost::any())); // add an empty traverser when this is being used as a start step.

    // For each traverser, a traverser should be created for each Vertex and passed to the next step
    CPUGraph* bg = static_cast<CPUGraph*>(trv->getGraph());

    
    std::vector<boost::any> element_ids = this->get_element_ids();

    // Short circuit if there are no element ids.
    // Gets ALL elements.
    if(element_ids.empty()) {
        std::vector<Vertex*>& vertices = bg->access_vertices();
        TraverserSet new_traversers;
        std::for_each(traversers.begin(), traversers.end(), [&](Traverser& t) {
            for(int k = 0; k < bg->numVertices(); ++k) new_traversers.push_back(Traverser(vertices[k]));
        });

        traversers.swap(new_traversers);

        return;
    }

    // Make map of element ids to counts
    std::unordered_map<uint64_t, unsigned long> element_id_counts;
    std::for_each(element_ids.begin(), element_ids.end(), [&](boost::any id_ctr) {
        uint64_t id = boost::any_cast<uint64_t>(id_ctr);
        if(element_id_counts.find(id) != element_id_counts.end()) element_id_counts[id]++;
        else element_id_counts[id] = 1;
    });

    TraverserSet new_traversers;
    // For each traverser...
    std::for_each(traversers.begin(), traversers.end(), [&](Traverser& trv) {
        std::for_each(element_ids.begin(), element_ids.end(), [&](boost::any id_ctr) {
            Vertex* v = bg->get_vertex(id_ctr);
            uint64_t id = boost::any_cast<uint64_t>(id_ctr);
            for(auto k = 0; k < element_id_counts[id]; k++) new_traversers.push_back(Traverser(v));
            //TODO retain side effects, path
        });
    });

    traversers.swap(new_traversers);
}