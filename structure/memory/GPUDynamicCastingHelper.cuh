#pragma once

#include <inttypes.h>
#include "gremlinxx/gremlinxx.h"
#include <thrust/functional.h>

namespace bitgraph {
    namespace memory {
        template<typename T, typename U>
        struct caster : public thrust::unary_function<T, U>{
            __host__ __device__
            U operator()(T t) const {
                return static_cast<U>(t);
            }
        };

        /*
            Copies casted data from the input array into the
            output array.
        */
        template<typename T, typename U>
        void array_cast(T* input_array, U* output_array, size_t array_length);

        template<typename T>
        void
        array_cast_inner_wrapper(T* input_array, void* output_array, size_t array_length, gremlinxx::comparison::C output_dtype);

        void
        array_cast_outer_wrapper(void* input_array, void* output_array, size_t array_length, gremlinxx::comparison::C input_dtype, gremlinxx::comparison::C output_dtype);

        gremlinxx::comparison::C get_output_dtype(gremlinxx::comparison::C left_dtype, gremlinxx::comparison::C right_dtype);

        /*
            Appends the elements in right to the end of left, upcasting dtype if necessary.
            Frees left and right.
        */
        std::tuple<void*, gremlinxx::comparison::C, size_t> device_array_combine(void* left, gremlinxx::comparison::C left_dtype, size_t left_length, void* right, gremlinxx::comparison::C right_dtype, size_t right_length);

    }       
}