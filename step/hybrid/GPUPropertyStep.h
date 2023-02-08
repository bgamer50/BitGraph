#pragma once

#define GPU_PROPERTY_STEP 0x21

#include <tuple>

#include "step/TraversalStep.h"
#include "step/gpu/GPUTraversalHelper.h"
#include "structure/memory/GPUPropertyTable.h"
#include "structure/memory/GPUDynamicCastingHelper.h"

/**
    Gathers the properties for a particular Vertex.
    Only supports "VALUE" property steps currently.
**/
class GPUPropertyStep: public TraversalStep {
    private:
        std::vector<std::string> keys; //duplicates are allowed, per api

    public:
        GPUPropertyStep(std::vector<std::string> keys) : TraversalStep(MAP, GPU_PROPERTY_STEP) {
            this->keys = keys;
        }

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers) {
            // check the dtype; autopromote to highest required type (i.e. int64, uint64 -> int64; int32, float32 -> float32)
            // later on, the min steps can force a type conversion as well 
            // (there is probably a good way to do this on GPU)
            
            GPUGraph* graph = static_cast<GPUGraph*>(traversal->getTraversalSource()->getGraph());
            bitgraph::memory::GPUPropertyTable& property_table = graph->access_property_table();

            // Get values for the bound traversers...
            for(Traverser& trv : traversers) {
                if(trv.get().type() != typeid(gpu_traverser_info_t)) {
                    throw std::runtime_error("GPU Property Access Should be done through GPU traversers!");
                }

                gpu_traverser_info_t traverser_info = boost::any_cast<gpu_traverser_info_t>(trv.get());
                if(traverser_info.traverser_dtype != gremlinxx::comparison::C::VERTEX) {
                    throw std::runtime_error("Traverser dtype was not of vertex type!");
                }

                gremlinxx::comparison::C current_dtype = property_table.get_dtype(
                    this->keys.front()
                );
                
                size_t* originating_indices;
                void* current_values;
                size_t num_current_values;
                std::tie(
                    originating_indices,
                    current_values,
                    num_current_values
                    ) = property_table.get_property_values_device(
                        static_cast<size_t*>(traverser_info.traversers), // traversers over vertices are actually gpu vertex ids (NOT pointers)
                        traverser_info.num_traversers,
                        this->keys.front(),
                        false // traversal semantics imply a non-strict search (it is ok if not all vertices have this value)
                );
                cudaDeviceSynchronize();
                cudaCheckErrors("GPUPropertyStep: get values for first key");

                for(auto it = this->keys.begin() + 1; it < this->keys.end(); ++it) {
                    std::string property_key = *it;
                    auto next_dtype = property_table.get_dtype(property_key);
                    
                    size_t* next_originating_indices;
                    void* next_values;
                    size_t num_next_values;

                    std::tie(
                        next_originating_indices,
                        next_values,
                        num_next_values
                        ) = property_table.get_property_values_device(
                            static_cast<size_t*>(traverser_info.traversers), // traversers over vertices are actually gpu vertex ids (NOT pointers)
                            traverser_info.num_traversers,
                            property_key,
                            false // traversal semantics imply a non-strict search (it is ok if not all vertices have this value)
                    );
                    cudaDeviceSynchronize();
                    cudaCheckErrors("GPUPropertyStep: get values for first key");

                    void* oi;
                    std::tie(oi, std::ignore, std::ignore) = bitgraph::memory::device_array_combine(
                        originating_indices,
                        gremlinxx::comparison::UINT64,
                        num_current_values,
                        next_originating_indices,
                        gremlinxx::comparison::UINT64,
                        num_next_values
                    );
                    originating_indices = static_cast<size_t*>(oi);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("GPUPropertyStep: update originating indices");

                    std::tie(current_values, current_dtype, num_current_values) = bitgraph::memory::device_array_combine(
                        current_values,
                        current_dtype,
                        num_current_values,
                        next_values,
                        next_dtype,
                        num_next_values
                    );
                    cudaDeviceSynchronize();
                    cudaCheckErrors("GPUPropertyStep: update curent values");
                }

                traverser_info.traversers = current_values;
                traverser_info.num_traversers = num_current_values;
                traverser_info.traverser_dtype = current_dtype;
                traverser_info.paths.push_back(
                    std::make_pair(
                        originating_indices,
                        num_current_values       
                    )
                );
            }
        }

        using TraversalStep::getInfo;
        virtual std::string getInfo() {
            return "GPUPropertyStep{}";
        }

};
