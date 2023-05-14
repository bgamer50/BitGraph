#pragma once

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {
    namespace memory {
        template <typename T, typename U>
        std::pair<size_t, size_t> t_set_scatter(void* keys, void* new_keys, void* values, void* new_values, size_t* insert_indices, size_t* stencil, size_t N, size_t table_size);

        template<typename U>
        std::pair<size_t, size_t> set_scatter_dispatch_inner(void* keys, void* new_keys, gremlinxx::comparison::C keys_dtype, void* values, void* new_values, size_t* insert_indices, size_t* stencil, size_t N, size_t table_size);

        /*
            Sets the elements of the hash table (keys, values) for pairs in (new_keys, new_values)
            where the stencil value is true.
            Afterwards, removes the pairs that were added and returns the new number of entries
            in the hash table (it's possible some elements were overwritten rather than added)
            and the number of remaining elements.
        */
        std::pair<size_t, size_t> set_scatter(void* keys, void* new_keys, gremlinxx::comparison::C keys_dtype, void* values, void* new_values, gremlinxx::comparison::C values_dtype, size_t* insert_indices, size_t* stencil, size_t N, size_t table_size);
    }
}