#pragma once

#include "structure/memory/TypeErasure.cuh"

namespace bitgraph {
        namespace memory {

        void remove_if(TypeErasedVector& array, TypeErasedVector& stencil);

    }
}