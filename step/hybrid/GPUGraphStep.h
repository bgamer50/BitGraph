#ifndef GPU_GRAPH_STEP_H
#define GPU_GRAPH_STEP_H

#include <vector>
#include <boost/any.hpp>
#include "step/graph/GraphStep.h"


class GPUGraphStep: public GraphStep {
    private:
        bool start;

    public:
        GPUGraphStep(bool start, GraphStepType gs_type, std::vector<boost::any> element_ids)
        : GraphStep(gs_type, element_ids) {
            this->start = start;
        }

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers);
};

#include "structure/GPUGraph.h"
void GPUGraphStep::apply(GraphTraversal* trv, TraverserSet& traversers) {
            if(start) traversers.push_back(Traverser(boost::any())); // add an empty traverser when this is being used as a start step.
        
            // For each traverser, a traverser should be created for each Vertex and passed to the next step
            GPUGraph* gpu_graph = static_cast<GPUGraph*>(trv->getGraph());

            
            std::vector<boost::any> element_ids = this->get_element_ids();

            // Short circuit if there are no element ids.
            // Gets ALL elements.
            if(element_ids.empty()) {
                std::vector<Vertex*>& vertices = gpu_graph->access_vertices();
                size_t num_vertices = vertices.size();
                TraverserSet new_traversers;
                std::for_each(traversers.begin(), traversers.end(), [&](Traverser& t) {
                    for(int k = 0; k < num_vertices; ++k) new_traversers.push_back(Traverser(vertices[k]));
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
                    uint64_t id = boost::any_cast<uint64_t>(id_ctr);
                    Vertex* v = gpu_graph->get_vertex_with_cpu_id(id);
                    for(auto k = 0; k < element_id_counts[id]; k++) new_traversers.push_back(Traverser(v));
                    //TODO retain side effects, path
                });
            });

            traversers.swap(new_traversers);
        }

#endif