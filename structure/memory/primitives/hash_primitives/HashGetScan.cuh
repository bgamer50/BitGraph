#pragma once

#include "structure/memory/TypeErasure.cuh"
#include <cuda_runtime.h>

namespace bitgraph {
    namespace memory {
        // GET SCAN
        template <typename B>
        __global__ void k_get_scan(B* ignore_values, void* retrieved_values, void* keys, size_t key_size, void* values, size_t value_size, void* desired_keys, size_t table_size, size_t N_desired);

        TypeErasedVector get_scan(TypeErasedVector& keys, TypeErasedVector& values, TypeErasedVector& desired_keys);
    }
}