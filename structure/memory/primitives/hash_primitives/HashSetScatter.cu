#include "structure/memory/primitives/hash_primitives/HashSetScatter.cuh"

#include "structure/memory/GPUHashConstants.h"
#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {

        template <typename T, typename U>
        size_t t_set_scatter(void* keys, void* new_keys, void* values, void* new_values, size_t* insert_indices, size_t* stencil, size_t N) {
            auto zipped_kv = thrust::make_zip_iterator(
                thrust::make_tuple(
                    device_tptr_cast<T>(keys),
                    device_tptr_cast<U>(values)
                )
            );

            auto zipped_new_kv = thrust::make_zip_iterator(
                thrust::make_tuple(
                    device_tptr_cast<T>(new_keys),
                    device_tptr_cast<U>(new_values)
                )
            );

            thrust::scatter_if(
                thrust::device,
                zipped_new_kv,
                zipped_new_kv + N,
                thrust::device_pointer_cast<size_t>(insert_indices),
                thrust::device_pointer_cast<size_t>(stencil),
                zipped_kv
            );

            auto new_end = thrust::remove_if(
                thrust::device,
                zipped_new_kv,
                zipped_new_kv + N,
                thrust::device_pointer_cast<size_t>(stencil),
                thrust::identity<size_t>()
            );

            return new_end - zipped_new_kv;
        }

        template<typename U>
        size_t set_scatter_dispatch_inner(void* keys, void* new_keys, gremlinxx::comparison::C keys_dtype, void* values, void* new_values, size_t* insert_indices, size_t* stencil, size_t N) {
            switch(keys_dtype) {
                case gremlinxx::comparison::C::UINT32:
                    return t_set_scatter<uint32_t, U>(keys, new_keys, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::UINT64:
                    return t_set_scatter<uint64_t, U>(keys, new_keys, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::INT32:
                    return t_set_scatter<int32_t, U>(keys, new_keys, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::INT64:
                    return t_set_scatter<int64_t, U>(keys, new_keys, values, new_values, insert_indices, stencil, N);
            }

            throw std::runtime_error("Invalid value type for hash table!");
        }
    
        /*
            Sets the elements of the hash table (keys, values) for pairs in (new_keys, new_values)
            where the stencil value is true.
            Afterwards, removes the pairs that were added and returns the number of pairs in
            (new_keys, new_values) that are remaining (the new length of that array).
        */
        size_t set_scatter(void* keys, void* new_keys, gremlinxx::comparison::C keys_dtype, void* values, void* new_values, gremlinxx::comparison::C values_dtype, size_t* insert_indices, size_t* stencil, size_t N) {
            switch(values_dtype) {
                case gremlinxx::comparison::C::UINT8:
                    return set_scatter_dispatch_inner<uint8_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::UINT32:
                    return set_scatter_dispatch_inner<uint32_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::UINT64:
                    return set_scatter_dispatch_inner<uint64_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::INT8:
                    return set_scatter_dispatch_inner<int8_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::INT32:
                    return set_scatter_dispatch_inner<int32_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::INT64:
                    return set_scatter_dispatch_inner<int64_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::FLOAT64:
                    return set_scatter_dispatch_inner<double>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::FLOAT32:
                    return set_scatter_dispatch_inner<float>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
                case gremlinxx::comparison::C::STRING:
                    return set_scatter_dispatch_inner<uint64_t>(keys, new_keys, keys_dtype, values, new_values, insert_indices, stencil, N);
            }

            throw std::runtime_error("Invalid value type for hash table!");
        }

    }
}