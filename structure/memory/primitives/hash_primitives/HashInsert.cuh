#pragma once

#include <cuda_runtime.h>

namespace bitgraph {
    namespace memory {
        /*
            Type-erased primitive that determines where new elements of a hash table should be inserted.
        */
        __global__ void k_hash_insert(size_t* insert_indices, void* keys, void* keys_new, size_t key_size, size_t table_size, size_t N_new);

        /*
            Returns the indices in the hash table where the provided new elements should be inserted.
            Collisions are possible.
        */
        size_t* hash_insert(void* keys, void* keys_new, size_t key_size, size_t table_size, size_t N_new);

    }
}