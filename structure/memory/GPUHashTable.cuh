#pragma once

#include "gremlinxx/gremlinxx.h"

#include "structure/memory/TypeErasure.cuh"

namespace bitgraph {
    namespace memory {
        
        class GPUHashTable {
            private:
                TypeErasedVector keys;
                TypeErasedVector values;
                size_t num_elements_in_table;
                float max_fill_factor = 0.75;
            
            public:
                GPUHashTable(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C key_dtype, gremlinxx::comparison::C val_dtype, size_t reserved_size=0);

                GPUHashTable();

                /*
                    This method sets the given keys to the given values.
                */
                void set(TypeErasedVector& new_keys, TypeErasedVector& new_values);

                std::pair<TypeErasedVector, TypeErasedVector> get(TypeErasedVector& desired_keys, bool strict=true, bool return_indices=false);

                void resize(size_t new_table_size);

                inline size_t size() {
                    return this->num_elements_in_table;
                }

                inline size_t table_size() {
                    return this->keys.size();
                }

                inline gremlinxx::comparison::C get_key_dtype() {
                    return this->keys.get_dtype();
                }

                inline gremlinxx::comparison::C get_value_dtype() {
                    return this->values.get_dtype();
                }
        };

    }
}