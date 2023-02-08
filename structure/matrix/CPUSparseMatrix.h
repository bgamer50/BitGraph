#pragma once

#include <vector>
#include <optional>

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
        sparse_matrix_host<T> sparse_make(size_t r, size_t c) {
            sparse_matrix_host<T> M;
            M.nnz = 0;
            M.row_ptr.resize(r+1, 0);
            M.num_rows = r;
            M.num_cols = c;
            return M;
        }

        template<typename T>
        T sparse_get(sparse_matrix_host<T>& M, size_t i, size_t j) {
            size_t rs = M.row_ptr[i];
            size_t re = M.row_ptr[i+1];
            for(size_t k = rs; k < re; ++k) {
                if(j == M.col_ptr[k]) return M.values[k];
            }

            return 0.0;
        }

        template <typename T>
        void sparse_set(sparse_matrix_host<T>& M, size_t i, size_t j, T val) {
            size_t rs = M.row_ptr[i];
            size_t re = M.row_ptr[i+1];

            for(size_t k = rs; k < re; ++k) {
                if(j == M.col_ptr[k]) { 
                    if(val != 0.0) M.values[k] = val;
                    else {
                        M.nnz -= 1;
                        M.col_ptr.erase(M.col_ptr.begin() + k);
                        M.values.erase(M.values.begin() + k);
                        for(size_t r = i + 1; r < M.row_ptr.size(); ++r) M.row_ptr[r] -= 1;
                    }

                    return;
                }
            }

            if(rs == re) {
                M.col_ptr.insert(M.col_ptr.begin() + rs, j);
                M.values.insert(M.values.begin() + rs, val);
            }
            else {
                size_t c = rs;
                while(c < M.col_ptr.size() && M.col_ptr[c] < j) ++c;
                M.col_ptr.insert(M.col_ptr.begin() + c, j);
                M.values.insert(M.values.begin() + c, val);
            }

            M.nnz += 1;
            for(size_t r = i + 1; r < M.row_ptr.size(); ++r) M.row_ptr[r] += 1;
        }
    }
}
