#include "structure/matrix/CPUSparseMatrix.h"
#include <inttypes.h>
#include <optional>

namespace bitgraph {
    namespace matrix {
        template<>
        struct sparse_matrix_host<uint64_t>;
        template<>
        struct sparse_matrix_host<uint32_t>;
        template<>
        struct sparse_matrix_host<uint8_t>;
        template<>
        struct sparse_matrix_host<int64_t>;
        template<>
        struct sparse_matrix_host<int32_t>;
        template<>
        struct sparse_matrix_host<int8_t>;
        template<>
        struct sparse_matrix_host<float>;
        template<>
        struct sparse_matrix_host<double>;

        // -------------------------------------------------------------------------------------

        template <typename T>
        sparse_matrix_host<T> sparse_make(size_t r, size_t c) {
            sparse_matrix_host<T> M;
            M.nnz = 0;
            M.row_ptr.resize(r+1, 0);
            M.num_rows = r;
            M.num_cols = c;
            return M;
        }

        template<>
        sparse_matrix_host<uint64_t> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<uint32_t> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<uint8_t> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<int64_t> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<int32_t> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<int8_t> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<float> sparse_make(size_t r, size_t c);
        template<>
        sparse_matrix_host<double> sparse_make(size_t r, size_t c);

        // -------------------------------------------------------------------------------------

        template<typename T>
        T sparse_get(sparse_matrix_host<T>& M, size_t i, size_t j) {
            size_t rs = M.row_ptr[i];
            size_t re = M.row_ptr[i+1];
            for(size_t k = rs; k < re; ++k) {
                if(j == M.col_ptr[k]) return M.values[k];
            }

            return 0.0;
        }

        template<>
        uint64_t sparse_get(sparse_matrix_host<uint64_t>& M, size_t i, size_t j);
        template<>
        uint32_t sparse_get(sparse_matrix_host<uint32_t>& M, size_t i, size_t j);
        template<>
        uint8_t sparse_get(sparse_matrix_host<uint8_t>& M, size_t i, size_t j);
        template<>
        int64_t sparse_get(sparse_matrix_host<int64_t>& M, size_t i, size_t j);
        template<>
        int32_t sparse_get(sparse_matrix_host<int32_t>& M, size_t i, size_t j);
        template<>
        int8_t sparse_get(sparse_matrix_host<int8_t>& M, size_t i, size_t j);
        template<>
        float sparse_get(sparse_matrix_host<float>& M, size_t i, size_t j);
        template<>
        double sparse_get(sparse_matrix_host<double>& M, size_t i, size_t j);

        // -------------------------------------------------------------------------------------

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

        template<>
        void sparse_set(sparse_matrix_host<uint64_t>& M, size_t i, size_t j, uint64_t val);
        template<>
        void sparse_set(sparse_matrix_host<uint32_t>& M, size_t i, size_t j, uint32_t val);
        template<>
        void sparse_set(sparse_matrix_host<uint8_t>& M, size_t i, size_t j, uint8_t val);
        template<>
        void sparse_set(sparse_matrix_host<int64_t>& M, size_t i, size_t j, int64_t val);
        template<>
        void sparse_set(sparse_matrix_host<int32_t>& M, size_t i, size_t j, int32_t val);
        template<>
        void sparse_set(sparse_matrix_host<int8_t>& M, size_t i, size_t j, int8_t val);
        template<>
        void sparse_set(sparse_matrix_host<float>& M, size_t i, size_t j, float val);
        template<>
        void sparse_set(sparse_matrix_host<double>& M, size_t i, size_t j, double val);

    }
}