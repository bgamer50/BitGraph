#pragma once

#include "structure/memory/TypeErasure.cuh"

namespace bitgraph {
    namespace memory {
        
        template<typename T, typename U>
        void t_filter_valid_values(TypeErasedVector& keys, TypeErasedVector& values);

        template<typename T>
        void filter_valid_values_dispatch_inner(TypeErasedVector& keys, TypeErasedVector& values);

        void filter_valid_values(TypeErasedVector& keys, TypeErasedVector& values);
        
    }
}
