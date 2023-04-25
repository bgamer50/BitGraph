#pragma once

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {
    namespace memory {
        template<typename T, typename U>
        void t_set_sort(size_t* sort_keys, void* keys, void* values, size_t N);

        template <typename U>
        void set_sort_dispatch_inner(size_t* sort_keys, void* keys, gremlinxx::comparison::C keys_dtype, void* values, size_t N);

        /*
            Sorts the kv pairs given by elements of (keys, values) by the given sort keys.
            So for each triple (sk, k, v) the elements are reordered by sk (including sk).
        */
        void set_sort(size_t* sort_keys, void* keys, gremlinxx::comparison::C keys_dtype, void* values, gremlinxx::comparison::C values_dtype, size_t N);
    }
}