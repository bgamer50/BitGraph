#include "step/hybrid/GPUPropertyStep.cuh"
#include "step/gpu/GPUTraversalHelper.cuh"
#include "structure/memory/GPUPropertyTable.cuh"
#include "structure/memory/GPUDynamicCastingHelper.cuh"
#include "structure/GPUGraph.cuh"
#include <tuple>

GPUPropertyStep::GPUPropertyStep(std::vector<std::string> keys) : TraversalStep(MAP, GPU_PROPERTY_STEP) {
    this->keys = keys;
}

std::string GPUPropertyStep::getInfo() {
    return "GPUPropertyStep{}";
}

void GPUPropertyStep::apply(GraphTraversal* traversal, TraverserSet& traversers) {
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

        gremlinxx::comparison::C first_dtype = property_table.get_dtype(
            this->keys.front()
        );
        
        auto elements_of_interest = TypeErasedVector(
            bitgraph::memory::memory_type::DEVICE,
            gremlinxx::comparison::C::UINT64,
            traverser_info.traversers,
            traverser_info.num_traversers,
            false
        );
        elements_of_interest.name = "elements of interest";

        std::cout << "about to query property table" << std::endl;
        TypeErasedVector current_values;
        TypeErasedVector originating_indices;
        std::tie(current_values, originating_indices) = property_table.get_property_values(
            this->keys.front(),
            elements_of_interest,
            false,
            true
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("get property values");

        cudaDeviceSynchronize();
        cudaCheckErrors("GPUPropertyStep: get values for first key");

        for(auto it = this->keys.begin() + 1; it < this->keys.end(); ++it) {
            std::string property_key = *it;
            auto next_dtype = property_table.get_dtype(property_key);
            
            TypeErasedVector next_originating_indices;
            TypeErasedVector next_values;

            std::tie(next_values, next_originating_indices) = property_table.get_property_values(
                property_key,
                elements_of_interest,
                false,
                true
            );

            cudaDeviceSynchronize();
            cudaCheckErrors("GPUPropertyStep: get values for additional key");

            originating_indices.insert(originating_indices.size(), next_originating_indices);
            next_originating_indices.clear();

            cudaDeviceSynchronize();
            cudaCheckErrors("GPUPropertyStep: update originating indices");

            void* current_values_data = current_values.data();
            size_t num_current_values = current_values.size();
            gremlinxx::comparison::C current_dtype = current_values.get_dtype();
            current_values.disown(); // device_array_combine does its own deletion

            void* next_values_data = next_values.data();
            size_t num_next_values = next_values.size();
            next_values.disown(); // device_array_combine does its own deletion
            
            std::tie(current_values_data, current_dtype, num_current_values) = bitgraph::memory::device_array_combine(
                current_values_data,
                current_dtype,
                num_current_values,
                next_values_data,
                next_dtype,
                num_next_values
            );
            cudaDeviceSynchronize();
            cudaCheckErrors("GPUPropertyStep: update current values");

            current_values = TypeErasedVector(
                bitgraph::memory::memory_type::DEVICE,
                current_dtype,
                current_values_data,
                num_current_values,
                false
            );
        }

        traverser_info.traversers = current_values.data();
        traverser_info.num_traversers = current_values.size();
        traverser_info.traverser_dtype = current_values.get_dtype();
        traverser_info.paths.push_back(
            std::make_pair(
                static_cast<size_t*>(originating_indices.data()),
                current_values.size()
            )
        );

        current_values.disown();
        originating_indices.disown();

        trv.replace_data(traverser_info);
    }
}