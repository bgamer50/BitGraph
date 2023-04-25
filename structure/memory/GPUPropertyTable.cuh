#pragma once

#include <cstdint>
#include <boost/any.hpp>
#include <sstream>

#include <unordered_map>
#include <cuda_runtime.h>
#include <thrust/functional.h>

#include "structure/memory/StringUtils.h"
#include "structure/memory/TypeErasure.cuh"
#include "structure/memory/GPUHashTable.cuh"

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {
    namespace memory {
        
        /*
            Wraps an unordered map of property index objects.
        */
        class GPUPropertyTable {
            private:
                std::unordered_map<std::string, GPUHashTable> key_to_index_map;
            public:
                void declare_property(std::string property_name, gremlinxx::comparison::C property_dtype, size_t initial_max_size);

                void set_property_values(std::string property_name, TypeErasedVector& elements, TypeErasedVector& values, bool strict=true);

                void get_property_values(std::string property_name, TypeErasedVector& elements, TypeErasedVector& values, bool strict=true);

                inline bool has_property_key(std::string property_name) {
                    return this->key_to_index_map.find(property_name) != this->key_to_index_map.end();
                }

                inline gremlinxx::comparison::C get_dtype(std::string property_name) {
                    if(!has_property_key(property_name)) {
                        std::stringstream sx;
                        sx << "Invalid property key " << property_name;
                        throw std::runtime_error(sx.str());
                    }

                    return this->key_to_index_map[property_name].get_value_dtype();
                }
        };
    }
}