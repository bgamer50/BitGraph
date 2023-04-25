#include "structure/memory/primitives/hash_primitives/HashFilterValid.cuh"

#include "structure/memory/GPUHashConstants.h"
#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {
        template<typename T, typename U>
        void t_filter_valid_values(TypeErasedVector& keys, TypeErasedVector& values) {
            thrust::remove_if(
                device_tptr_cast<U>(values.data()),
                device_tptr_cast<U>(values.data()) + values.size(),
                device_tptr_cast<T>(keys.data()),
                bitgraph::memory::is_max_val<T>()
            );

            auto new_end = thrust::remove(
                device_tptr_cast<T>(keys.data()),
                device_tptr_cast<T>(keys.data()) + keys.size(),
                std::numeric_limits<T>::max()
            );

            size_t new_size = new_end - device_tptr_cast<T>(keys.data());
            keys.resize(new_size);
            values.resize(new_size);
        }

        template<typename T>
        void filter_valid_values_dispatch_inner(TypeErasedVector& keys, TypeErasedVector& values) {
            switch(values.get_dtype()) {
                case gremlinxx::comparison::C::UINT8:
                    return t_filter_valid_values<T, uint8_t>(keys, values);
                case gremlinxx::comparison::C::UINT32:
                    return t_filter_valid_values<T, uint32_t>(keys, values);
                case gremlinxx::comparison::C::UINT64:
                    return t_filter_valid_values<T, uint64_t>(keys, values);
                case gremlinxx::comparison::C::INT8:
                    return t_filter_valid_values<T, int8_t>(keys, values);
                case gremlinxx::comparison::C::INT32:
                    return t_filter_valid_values<T, int32_t>(keys, values);
                case gremlinxx::comparison::C::INT64:
                    return t_filter_valid_values<T, int64_t>(keys, values);
                case gremlinxx::comparison::C::FLOAT64:
                    return t_filter_valid_values<T, double>(keys, values);
                case gremlinxx::comparison::C::FLOAT32:
                    return t_filter_valid_values<T, float>(keys, values);
                case gremlinxx::comparison::C::STRING:
                    return t_filter_valid_values<T, uint64_t>(keys, values);
            }

            throw std::runtime_error("Invalid valid data type for hash table!");
        }

        void filter_valid_values(TypeErasedVector& keys, TypeErasedVector& values) {
            switch(keys.get_dtype()) {
                case gremlinxx::comparison::C::UINT32:
                    return filter_valid_values_dispatch_inner<uint32_t>(keys, values);
                case gremlinxx::comparison::C::UINT64:
                    return filter_valid_values_dispatch_inner<uint64_t>(keys, values);
                case gremlinxx::comparison::C::INT32:
                    return filter_valid_values_dispatch_inner<int32_t>(keys, values);
                case gremlinxx::comparison::C::INT64:
                    return filter_valid_values_dispatch_inner<int64_t>(keys, values);
            }

            throw std::runtime_error("Invalid key data type for hash table!");
        
        }

    }
}