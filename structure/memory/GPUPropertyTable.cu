#include "structure/memory/GPUPropertyTable.cuh"

#include <boost/core/typeinfo.hpp>
#include "util/cuda_utils.cuh"
#include <cuda_runtime.h>

#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {

        void GPUPropertyTable::declare_property(std::string property_name, gremlinxx::comparison::C property_dtype, size_t initial_max_size) {
            if(property_dtype == gremlinxx::comparison::C::VERTEX) {
                throw std::runtime_error("Cannot store a vertex as a property, consider using id() instead");
            }

            this->key_to_index_map[property_name] = GPUPropertyIndex(
                initial_max_size,
                property_dtype
            );
        }

        void GPUPropertyTable::set_property_value(uint64_t element, std::string property_name,  boost::any property_value) {
            this->key_to_index_map[property_name].set(element, property_value);
        }

        boost::any GPUPropertyTable::get_property_value(uint64_t element, std::string property_name) {
            std::vector<uint64_t> vec;
            vec.push_back(element);
            std::vector<boost::any> found_values = this->key_to_index_map[property_name].get(vec);
            if(found_values.empty()) {
                std::stringstream sx;
                sx << "No value was found for property " << property_name << " on vertex " << element;
                throw std::runtime_error(sx.str());
            }

            return found_values.front();
        }

        /*
            Gets the values for the given property on the given elements.
            If strict = true (default) then an exception will be thrown if one or more elements are missing the property.
            If strict = false then however many values were found will be returned.
        */
        std::vector<boost::any> GPUPropertyTable::get_property_values(std::vector<uint64_t> elements, std::string property_name, bool strict) {
            if(!has_property_key(property_name)) {
                std::stringstream sx;
                sx << "Invalid property key " << property_name;
                throw std::runtime_error(sx.str());
            }

            auto found_values = this->key_to_index_map[property_name].get(elements);
            if(strict && found_values.size() < elements.size()) {
                std::stringstream sx;
                sx << "Some elements were missing a value for the property " << property_name;
                throw std::runtime_error(sx.str());
            }

            return found_values;
        }

        /*
            Returns a tuple of originating index, property value, num values
        */
        std::tuple<size_t*, void*, size_t> GPUPropertyTable::get_property_values_device(uint64_t* desired_element_ids, size_t N_desired, std::string property_name, bool strict) {
            if(!has_property_key(property_name)) {
                std::stringstream sx;
                sx << "Invalid property key " << property_name;
                throw std::runtime_error(sx.str());
            }

            auto result = this->key_to_index_map[property_name].get_device(desired_element_ids, N_desired);
            if(strict && std::get<2>(result) < N_desired) {
                std::stringstream sx;
                sx << "Some elements were missing a value for the property " << property_name;
                throw std::runtime_error(sx.str());
            }

            return result;
        }


    }
}