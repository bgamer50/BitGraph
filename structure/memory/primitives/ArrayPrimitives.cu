#include "structure/memory/primitives/ArrayPrimitives.cuh"

#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {

        template <typename T, typename U>
        size_t t_remove_if(TypeErasedVector& array, TypeErasedVector& stencil, size_t N) {
            auto new_last = thrust::remove_if(
                thrust::device,
                device_tptr_cast<T>(array.data()),
                device_tptr_cast<T>(array.data()) + N,
                device_tptr_cast<U>(stencil.data()),
                thrust::identity<U>()
            );

            return new_last - device_tptr_cast<T>(array.data());
        
        }

        template<typename T>
        size_t remove_if_dispatch_inner(TypeErasedVector& array, TypeErasedVector& stencil, size_t N) {
            switch(stencil.get_dtype()) {
                case gremlinxx::comparison::C::UINT8:
                    return t_remove_if<T, uint8_t>(
                        array, 
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::UINT32:
                    return t_remove_if<T, uint32_t>(
                        array, 
                        stencil,
                        N
                );
                case gremlinxx::comparison::C::UINT64:
                    return t_remove_if<T, uint64_t>(
                        array, 
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::INT8:
                    return t_remove_if<T, int8_t>(
                        array, 
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::INT32:
                    return t_remove_if<T, int32_t>(
                        array, 
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::INT64:
                    return t_remove_if<T, int64_t>(
                        array, 
                        stencil,
                        N
                    );
            }

            throw std::runtime_error("Invalid dtype provided to remove_if");
        }

        size_t remove_if_dispatch_outer(TypeErasedVector& array, TypeErasedVector& stencil, size_t N) {
            switch(array.get_dtype()) {
                case gremlinxx::comparison::C::UINT8:
                    return remove_if_dispatch_inner<uint8_t>(
                        array,
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::UINT32:
                    return remove_if_dispatch_inner<uint32_t>(
                        array,
                        stencil,
                        N
                );
                case gremlinxx::comparison::C::UINT64:
                    return remove_if_dispatch_inner<uint64_t>(
                        array,
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::INT8:
                    return remove_if_dispatch_inner<int8_t>(
                        array,
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::INT32:
                    return remove_if_dispatch_inner<int32_t>(
                        array,
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::INT64:
                    return remove_if_dispatch_inner<int64_t>(
                        array,
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::FLOAT32:
                    return remove_if_dispatch_inner<float>(
                        array,
                        stencil,
                        N
                    );
                case gremlinxx::comparison::C::FLOAT64:
                    return remove_if_dispatch_inner<double>(
                        array,
                        stencil,
                        N
                    );
            }

            throw std::runtime_error("Invalid dtype provided to remove_if");
        }

    }
}