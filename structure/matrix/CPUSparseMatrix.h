#pragma once

#include <vector>
#include <inttypes.h>
#include <cstddef>

namespace bitgraph {
    namespace matrix {

        template<typename T>
        struct sparse_matrix_host {
            size_t nnz; // number of nonzero elements
            std::vector<T> values; // element values
            std::vector<size_t> row_ptr; // row pointer
            std::vector<size_t> col_ptr; // column pointer

            size_t num_rows;
            size_t num_cols;
        };

        template <typename T>
        sparse_matrix_host<T> sparse_make(size_t r, size_t c);

        template<typename T>
        T sparse_get(sparse_matrix_host<T>& M, size_t i, size_t j);

        template <typename T>
        void sparse_set(sparse_matrix_host<T>& M, size_t i, size_t j, T val);

    }
}
