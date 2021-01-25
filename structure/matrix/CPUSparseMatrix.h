#ifndef CPU_SPARSE_MATRIX_H
#define CPU_SPARSE_MATRIX_H

#include <vector>
#include <inttypes.h>

typedef struct sparse_matrix {
    int32_t nnz; // number of nonzero elements
    std::vector<float> values; // element values
    std::vector<int32_t> row_ptr; // row pointer
    std::vector<int32_t> col_ptr; // column pointer

    int32_t num_rows;
    int32_t num_cols;
} sparse_matrix_t;

sparse_matrix_t sparse_make(int32_t r, int32_t c) {
    sparse_matrix_t M;
    M.nnz = 0;
    M.row_ptr.resize(r+1, 0);
    M.num_rows = r;
    M.num_cols = c;
    return M;
}

float sparse_get(sparse_matrix_t& M, int32_t i, int32_t j) {
    int32_t rs = M.row_ptr[i];
    int32_t re = M.row_ptr[i+1];
    for(int32_t k = rs; k < re; ++k) {
        if(j == M.col_ptr[k]) return M.values[k];
    }

    return 0.0;
}

void sparse_set(sparse_matrix_t& M, int32_t i, int32_t j, float val) {
    int32_t rs = M.row_ptr[i];
    int32_t re = M.row_ptr[i+1];
    for(int32_t k = rs; k < re; ++k) {
        if(j == M.col_ptr[k]) { 
            if(val != 0.0) M.values[k] = val;
            else {
                M.nnz -= 1;
                M.col_ptr.erase(M.col_ptr.begin() + k);
                M.values.erase(M.values.begin() + k);
                for(int r = i + 1; r < M.row_ptr.size(); ++r) M.row_ptr[r] -= 1;
            }

            return;
        }
    }

    if(rs == re) {
        M.col_ptr.insert(M.col_ptr.begin() + rs, j);
        M.values.insert(M.values.begin() + rs, val);
    }
    else {
        int c = rs;
        while(M.col_ptr[c] < j) ++c;
        M.col_ptr.insert(M.col_ptr.begin() + c, j);
        M.values.insert(M.values.begin() + c, val);
    }

    M.nnz += 1;
    for(int r = i + 1; r < M.row_ptr.size(); ++r) M.row_ptr[r] += 1;
}

#endif