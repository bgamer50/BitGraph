#ifndef GPU_BIND_STEP_H
#define GPU_BIND_STEP_H

#define GPU_BIND_STEP 0x2e

#include "step/TraversalStep.h"
#include "step/gpu/GPUTraversalHelper.h"

#include <numeric>

class GPUBindStep : public TraversalStep {
    public:
        GPUBindStep() : TraversalStep(MAP, GPU_BIND_STEP) {}

        virtual void apply(GraphTraversal* parent_traversal, TraverserSet& traversers) {
            std::cout << "vertices:" << std::endl;
            for(Traverser& trv : traversers) std::cout << boost::any_cast<uint64_t>(boost::any_cast<Vertex*>(trv.get())->id()) << std::endl;
            std::cout << "bind " << traversers.size() << " traversers" << std::endl;
            gpu_traverser_info_t traverser_info;
            traverser_info.traversers = to_gpu(traversers);

            int32_t* originating_traversers;
            cudaMalloc((void**) &originating_traversers, sizeof(int32_t) * traversers.size());
            cudaDeviceSynchronize();
            cudaCheckErrors("allocate originating traversers");

            std::vector<int32_t> h_originating_traversers(traversers.size());
            std::iota(h_originating_traversers.begin(), h_originating_traversers.end(), 0);
            cudaMemcpy(originating_traversers, h_originating_traversers.data(), sizeof(int32_t) * traversers.size(), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaCheckErrors("copy originating traversers");
            
            traverser_info.paths.push_back(std::make_pair(originating_traversers, traversers.size()));

            traverser_info.num_traversers = traversers.size();
            traverser_info.original_traversers.swap(traversers);

            traversers.push_back(Traverser(traverser_info));
        }

        virtual std::string getInfo() {
            return "GPUBindStep{}";
        }
};

#endif