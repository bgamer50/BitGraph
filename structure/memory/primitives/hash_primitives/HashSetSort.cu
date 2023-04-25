#include "structure/memory/primitives/hash_primitives/HashSetSort.cuh"

#include "structure/memory/GPUHashConstants.h"
#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {

        template<typename T, typename U>
        void t_set_sort(size_t* sort_keys, void* keys, void* values, size_t N) {
            auto zipped_kv = thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::device_pointer_cast<T>(static_cast<T*>(keys)),
                    thrust::device_pointer_cast<U>(static_cast<U*>(values))
                )
            );

            thrust::sort_by_key(
                thrust::device,
                thrust::device_pointer_cast<size_t>(sort_keys),
                thrust::device_pointer_cast<size_t>(sort_keys) + N,
                zipped_kv
            );
        }

        template <typename U>
        void set_sort_dispatch_inner(size_t* sort_keys, void* keys, gremlinxx::comparison::C keys_dtype, void* values, size_t N) {
            switch(keys_dtype) {
                case gremlinxx::comparison::C::UINT32:
                    return t_set_sort<uint32_t, U>(sort_keys, keys, values, N);
                case gremlinxx::comparison::C::UINT64:
                    return t_set_sort<uint64_t, U>(sort_keys, keys, values, N);
                case gremlinxx::comparison::C::INT32:
                    return t_set_sort<int32_t, U>(sort_keys, keys, values, N);
                case gremlinxx::comparison::C::INT64:
                    return t_set_sort<int64_t, U>(sort_keys, keys, values, N);
            }

            throw std::runtime_error("Invalid key data type for hash table!");
        }

        /*
            Sorts the kv pairs given by elements of (keys, values) by the given sort keys.
            So for each triple (sk, k, v) the elements are reordered by sk (including sk).
        */
        void set_sort(size_t* sort_keys, void* keys, gremlinxx::comparison::C keys_dtype, void* values, gremlinxx::comparison::C values_dtype, size_t N) {
            switch(values_dtype) {
                case gremlinxx::comparison::C::UINT8:
                    return set_sort_dispatch_inner<uint8_t>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::UINT32:
                    return set_sort_dispatch_inner<uint32_t>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::UINT64:
                    return set_sort_dispatch_inner<uint64_t>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::INT8:
                    return set_sort_dispatch_inner<int8_t>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::INT32:
                    return set_sort_dispatch_inner<int32_t>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::INT64:
                    return set_sort_dispatch_inner<int64_t>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::FLOAT64:
                    return set_sort_dispatch_inner<double>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::FLOAT32:
                    return set_sort_dispatch_inner<float>(sort_keys, keys, keys_dtype, values, N);
                case gremlinxx::comparison::C::STRING:
                    return set_sort_dispatch_inner<uint64_t>(sort_keys, keys, keys_dtype, values, N);
            }

            throw std::runtime_error("Invalid value data type for hash table!");
        }

    }
}