#pragma once

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/adjacent_difference.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>

#include <thrust/unique.h>

#include <thrust/sort.h>

#include <thrust/reduce.h>

#include <thrust/execution_policy.h>

#include <limits>

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

        template<typename T>
        struct is_max_val {
            __host__ __device__ bool operator() (const T val) {
                return val == std::numeric_limits<T>::max();
            }
        };

    }
}