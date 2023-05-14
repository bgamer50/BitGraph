#include "structure/memory/GPUPropertyTable.cuh"

#include <boost/core/typeinfo.hpp>
#include "util/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <sstream>

#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {

        void GPUPropertyTable::declare_property(std::string property_name, gremlinxx::comparison::C property_dtype, size_t initial_max_size, bitgraph::memory::memory_type memory_type) {
            if(property_dtype == gremlinxx::comparison::C::VERTEX) {
                throw std::runtime_error("Cannot store a vertex as a property, consider using id() instead");
            }

            this->key_to_index_map[property_name] = GPUHashTable(
                memory_type,
                gremlinxx::comparison::C::UINT64,
                property_dtype,
                initial_max_size
            );
        }

        void GPUPropertyTable::set_property_values(std::string property_name, TypeErasedVector& elements, TypeErasedVector& values) {
            auto table = this->key_to_index_map.find(property_name);
            if(table == this->key_to_index_map.end()) {
                std::stringstream sx;
                sx << "Property " << property_name << " does not exist!";
                throw std::runtime_error(sx.str());
            }

            table->second.set(
                elements,
                values
            );
        }

        std::pair<TypeErasedVector, TypeErasedVector> GPUPropertyTable::get_property_values(std::string property_name, TypeErasedVector& elements, bool strict, bool return_indices) {
            auto table = this->key_to_index_map.find(property_name);
            if(table == this->key_to_index_map.end()) {
                std::stringstream sx;
                sx << "Property " << property_name << " does not exist!";
                throw std::runtime_error(sx.str());
            }

            return table->second.get(
                elements,
                strict,
                return_indices
            );
        }

    }
}