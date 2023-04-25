#include "structure/memory/primitives/hash_primitives/HashGetScan.cuh"
#include "structure/memory/primitives/hash_primitives/HashBasic.cuh"

#include "structure/memory/GPUHashConstants.h"

#include "structure/memory/ThrustUtils.cuh"
#include "structure/memory/ArrayFunctions.cuh"

// GET SCAN
namespace bitgraph {
    namespace memory {  
        template <typename B>
        __global__ void k_get_scan(B* ignore_values, void* retrieved_values, void* keys, size_t key_size, void* values, size_t value_size, void* desired_keys, size_t table_size, size_t N_desired, size_t max_chain_iters, size_t max_total_iters) {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for(size_t k = index; k < N_desired; k += stride) {
                size_t key = 0;
                unsigned char* key_ptr = static_cast<unsigned char*>(desired_keys) + (key_size * k);
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

                if(existing_key == key) {
                    unsigned char* existing_value_ptr = static_cast<unsigned char*>(values) + (value_size * key_hash);
                    unsigned char* retrieved_values_ptr = static_cast<unsigned char*>(retrieved_values) + (index * value_size);
                    for(size_t b = 0; b < value_size; ++b) {
                        retrieved_values_ptr[b] = existing_value_ptr[b];
                    }

                    ignore_values[index] = false;
                } else {
                    ignore_values[index] = true;
                }
            }
        }

        TypeErasedVector get_scan(TypeErasedVector& keys, TypeErasedVector& values, TypeErasedVector& desired_keys) {
            TypeErasedVector retrieved_values = bitgraph::memory::make_vector_like(values, desired_keys.size());
            
            TypeErasedVector ignore_values(
                bitgraph::memory::memory_type::DEVICE,
                gremlinxx::comparison::C::INT8,
                desired_keys.size()
            );

            size_t key_size = gremlinxx::comparison::C_size[keys.get_dtype()];
            size_t value_size = gremlinxx::comparison::C_size[values.get_dtype()];

            size_t block_size = 128;
            size_t num_blocks = desired_keys.size() / block_size + 1;
            if(num_blocks > 1024) num_blocks = 1024;

            size_t table_size = keys.size();
            k_get_scan<char><<<block_size, num_blocks>>>(
                static_cast<char*>(ignore_values.data()),
                retrieved_values.data(),
                keys.data(),
                key_size,
                values.data(),
                value_size,
                desired_keys.data(),
                table_size,
                desired_keys.size(),
                bitgraph_get_max_permitted_chain_hash_iterations(table_size),
                bitgraph_get_max_permitted_hash_iterations(table_size)
            );

            bitgraph::memory::remove_if(retrieved_values, ignore_values);
            return retrieved_values;
        }

    }
}