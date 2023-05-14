#pragma once

#include <cstdint>
#include <boost/any.hpp>
#include <sstream>
#include <optional>
#include <utility>

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
                /*
                    Declares the property with the given name.  Properties are not limited to any particular element type.
                    Will overwrite an existing property with the same name.

                    Arguments
                    ---------
                    property_name: std::string
                        The name of the property
                    property_dtype: gremlinxx::comparison::C
                        The data type of the property
                    initial_max_size: size_t
                        The initial hash table size for the property being created.
                */
                void declare_property(
                    std::string property_name,
                    gremlinxx::comparison::C property_dtype,
                    size_t initial_max_size=32,
                    bitgraph::memory::memory_type memory_type=bitgraph::memory::memory_type::MANAGED
                );

                /*
                    For the provided elements, sets the property given by property_name to the provided values.

                    Arguments
                    ---------
                    property_name: std::string
                        The name of the property to set
                    elements: TypeErasedVector&
                        Reference to a type erased vector containing the elements whose property value is being set or updated.  Must have
                        the UINT64 data type.
                    values: TypeErasedVector&
                        Type erased vector containing the new property values for the given elements.  Must have the same dtype
                        as the value dtype for the property being set.

                */
                void set_property_values(std::string property_name, TypeErasedVector& elements, TypeErasedVector& values);

                /*
                    For the provided elements, get the value of the provided property.  Values are always returned ordered by
                    the original index of the elements requested.  This is true even if strict=true (missing values will just be omitted).

                    Arguments
                    ---------
                    property_name: std::string
                        The name of the property to get
                    elements: TypeErasedVector&
                        Reference to a type erased vector containing the elements of interest.
                    strict: bool
                        If true, will throw an exception of some elements do not have the provided property.  If false,
                        only the values for elements that have them will be returned.
                    return_indices: bool
                        If true, will return the original indices.  If strict=false, this argument is invalid and
                        an exception will be thrown.
                    
                    Returns
                    -------
                    std::pair<TypeErasedVector, TypeErasedVector>
                        first: The requested property values, as a TypeErasedVector.
                        second: The originating indices, as a TypeErasedVector, if return_indices was true (an empty vector otherwise).
                */
                std::pair<TypeErasedVector, TypeErasedVector> get_property_values(std::string property_name, TypeErasedVector& elements, bool strict=true, bool return_indices=false);

                /*
                    Check if there is a property with the given name.

                    Arguments
                    ---------
                    property_name: std::string
                        The name of the property to check.

                    Returns
                    -------
                    bool
                        true if the property exists, false otherwise.          
                */
                inline bool has_property_key(std::string property_name) {
                    return this->key_to_index_map.find(property_name) != this->key_to_index_map.end();
                }

                /*
                    Return the data type of a property.

                    Arguments
                    ---------
                    property_name: std::string
                        The name of the property to check.

                    Returns
                    -------
                    gremlinxx::comparison::C
                        The data type of the given property.
                */
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