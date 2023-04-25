#include "structure/memory/ArrayFunctions.cuh"

#include "structure/memory/primitives/ArrayPrimitives.cuh"

namespace bitgraph {
    namespace memory {
        void remove_if(TypeErasedVector& array, TypeErasedVector& stencil) {
            if(stencil.size() != array.size()) throw std::runtime_error("Mask size must match vector size!");
            size_t new_size = remove_if_dispatch_outer(array, stencil, array.size());
            array.resize(new_size);
        }
    }
}