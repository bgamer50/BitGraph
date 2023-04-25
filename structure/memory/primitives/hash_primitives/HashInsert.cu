#include "structure/memory/primitives/hash_primitives/HashInsert.cuh"
#include "structure/memory/primitives/hash_primitives/HashBasic.cuh"

#include "structure/memory/GPUHashConstants.h"

namespace bitgraph {
    namespace memory {

        /*
            Type-erased primitive that determines where new elements of a hash table should be inserted.
        */
        __global__ void k_hash_insert(size_t* insert_indices, void* keys, void* keys_new, size_t key_size, size_t table_size, size_t N_new, size_t max_chain_iters, size_t max_total_iters) {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for(size_t k = index; k < N_new; k += stride) {
                size_t key = 0;
                unsigned char* key_ptr = static_cast<unsigned char*>(keys_new) + (key_size * k);
                for(size_t b = 0; b < key_size; ++b) {
                    key <<= 8;
                    key |= key_ptr[b];
                }

                size_t key_hash = key;
                size_t existing_key;
                bool key_found = false;

                size_t iters = 0;
                do {
                    key_hash = bitgraph::memory::bitgraph_hash_fn(key_hash, table_size, iters, max_chain_iters);

                    existing_key = 0;
                    unsigned char* existing_key_ptr = static_cast<unsigned char*>(keys) + (key_size * key_hash);
                    for(size_t b = 0; b < key_size; ++b) {
                        existing_key <<= 8;
                        existing_key |= existing_key_ptr[b];
                        if(existing_key_ptr[b] != 0xff) key_found = true;
                    }

                    ++iters;
                } while(key_found && existing_key != key && iters < max_total_iters);

                insert_indices[k] = (iters < max_total_iters) ? key_hash : std::numeric_limits<size_t>::max();
            }
        }

        /*
            Returns the indices in the hash table where the provided new elements should be inserted.
            Collisions are possible.
        */
        size_t* hash_insert(void* keys, void* keys_new, size_t key_size, size_t table_size, size_t N_new) {
            size_t* insert_indices;
            cudaMalloc(&insert_indices, sizeof(size_t) * N_new);

            size_t block_size = 128;
            size_t num_blocks = N_new / block_size + 1;
            if(num_blocks > 1024) num_blocks = 1024;

            k_hash_insert<<<block_size, num_blocks>>>(
                insert_indices,
                keys,
                keys_new,
                key_size,
                table_size,
                N_new,
                bitgraph_get_max_permitted_chain_hash_iterations(table_size),
                bitgraph_get_max_permitted_hash_iterations(table_size)
            );

            return insert_indices;
        }

    }
}