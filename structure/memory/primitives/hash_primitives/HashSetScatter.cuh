#pragma once

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {
    namespace memory {
        template <typename T, typename U>
        size_t t_set_scatter(void* keys, void* new_keys, void* values, void* new_values, size_t* insert_indices, size_t* stencil, size_t N);

        template<typename U>
        size_t set_scatter_dispatch_inner(void* keys, void* new_keys, gremlinxx::comparison::C keys_dtype, void* values, void* new_values, size_t* insert_indices, size_t* stencil, size_t N);

        /*
            Sets the elements of the hash table (keys, values) for pairs in (new_keys, new_values)
            where the stencil value is true.
            Afterwards, removes the pairs that were added and returns the number of pairs in
            (new_keys, new_values) that are remaining (the new length of that array).
        */
        size_t set_scatter(void* keys, void* new_keys, gremlinxx::comparison::C keys_dtype, void* values, void* new_values, gremlinxx::comparison::C values_dtype, size_t* insert_indices, size_t* stencil, size_t N);
    }
}