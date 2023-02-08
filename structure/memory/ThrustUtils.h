#pragma once

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace bitgraph {
    namespace memory {

        struct plus_op : public thrust::unary_function<thrust::tuple<size_t,size_t>,size_t> {
            __device__ size_t operator()(thrust::tuple<size_t, size_t> t) const {
                return thrust::get<0>(t) + thrust::get<1>(t);
            }
        };

        struct minus_op : public thrust::unary_function<thrust::tuple<size_t,size_t>,size_t> {
            __device__ size_t operator()(thrust::tuple<size_t, size_t> t) const {
                return thrust::get<0>(t) - thrust::get<1>(t);
            }
        };

    }
}