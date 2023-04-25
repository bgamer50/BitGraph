#pragma once

#include "structure/memory/TypeErasure.cuh"

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {
    namespace memory {
        template <typename T, typename U>
        size_t t_remove_if(TypeErasedVector& array, TypeErasedVector& stencil, size_t N);

        template<typename T>
        size_t remove_if_dispatch_inner(TypeErasedVector& array, TypeErasedVector& stencil, size_t N);

        size_t remove_if_dispatch_outer(TypeErasedVector& array, TypeErasedVector& stencil, size_t N);

    }
}