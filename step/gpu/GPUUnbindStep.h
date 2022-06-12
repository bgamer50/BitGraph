#ifndef GPU_UNBIND_STEP_H
#define GPU_UNBIND_STEP_H

#define GPU_UNBIND_STEP 0x2f

#include "step/TraversalStep.h"

class GPUUnbindStep : public TraversalStep {
    private:
        bool free_gpu_memory = true;

    public:
        GPUUnbindStep() : TraversalStep(MAP, GPU_UNBIND_STEP) {}
        GPUUnbindStep(bool free_gpu_memory) : TraversalStep(MAP, GPU_UNBIND_STEP) { this->free_gpu_memory = free_gpu_memory; }

        virtual void apply(GraphTraversal* parent_traversal, TraverserSet& traversers) {
            //std::cout << "enter unbind step (free=" << this->free_gpu_memory << ")" << std::endl;

            TraverserSet new_traversers;
            for(Traverser& trv : traversers) {
                if(trv.get().type() == typeid(gpu_traverser_info_t)) {
                    gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(trv.get());
                    
                    // short-circuit if there are no traversers
                    if(traverser_info.num_traversers == 0) {
                        continue;
                    }

                    std::vector<int32_t> originating_traversers = collapse_path(traverser_info, true); // don't handle path info at the moment
                    std::vector<int32_t> new_traversed_vertices(traverser_info.num_traversers);
                    cudaMemcpy(new_traversed_vertices.data(), traverser_info.traversers, sizeof(int32_t) * traverser_info.num_traversers, cudaMemcpyDeviceToHost);

                    GPUGraph* gpu_graph = static_cast<GPUGraph*>(parent_traversal->getGraph());
                    
                    //std::cout << "num original traversers: " << traverser_info.original_traversers.size() << std::endl;
                    //for(int t : originating_traversers) std::cout << t << std::endl;
                    size_t old_size = new_traversers.size();
                    new_traversers.resize(old_size + traverser_info.num_traversers);
                    for(int k = 0; k < traverser_info.num_traversers; ++k) {
                        Vertex* v = static_cast<Vertex*>(gpu_graph->access_vertices()[new_traversed_vertices[k]]);
                        Traverser& originating_traverser = traverser_info.original_traversers[originating_traversers[k]];
                        
                        new_traversers[old_size + k].replace_data(v);
                        auto& se = originating_traverser.get_side_effects();
                        new_traversers[old_size + k].get_side_effects().insert(se.begin(), se.end());
                    }
                }
                else {
                    new_traversers.push_back(trv);
                }
            }
            
            traversers.swap(new_traversers);
            //std::cout << "leave unbind step " << std::endl;
        }

        virtual std::string getInfo() {
            return "GPUUnbindStep{}";
        }
};

#endif