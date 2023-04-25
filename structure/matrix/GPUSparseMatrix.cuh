#pragma once

#include <vector>
#include <string>


#include <cuda_runtime.h>

#include "gremlinxx/gremlinxx.h"
#include "structure/matrix/CPUSparseMatrix.h"

namespace bitgraph {
    namespace matrix {
        struct sparse_matrix_device {
            size_t num_rows;
            size_t num_cols;
            size_t nnz; // number of nonzero elements

            void* values = nullptr; // element values, device pointer, can be nullptr
            gremlinxx::comparison::C value_dtype = gremlinxx::comparison::C::FLOAT32;
            size_t* row_ptr; // row device pointer
            size_t* col_ptr; // column device pointer
        };

        /**
        * Converts a CSR sparse matrix on the host to a CSR matrix on the device. 
        **/
        template <typename T>
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<T>& host_matrix);

        /**
        * Returns the left index of the COO matrix.
        * The right index is the already-existing col ptr.
        */
        size_t* csr_to_coo(sparse_matrix_device& device_matrix);

        template <typename T>
        void zipped_sort_coo(size_t* src, size_t* dst, void* values, size_t num_edges);

        void zipped_sort_coo_helper(size_t* src, size_t* dst, void* values, gremlinxx::comparison::C values_dtype, size_t num_edges);

        /**
        * Transposes a CSR matrix on the device.
        **/
        sparse_matrix_device transpose_csr_matrix(sparse_matrix_device& device_matrix);


        inline sparse_matrix_device add_csr_matrices(sparse_matrix_device& device_matrix_A, sparse_matrix_device& device_matrix_B) {
            size_t* coo_A = csr_to_coo(device_matrix_A);
            size_t* coo_B = csr_to_coo(device_matrix_B);

            throw std::runtime_error("Adding csr matrices currently unimplemented");
        }

        /**
        * Converts a CSR matrix on the device to a CSR sparse matrix on the host. 
        **/
        template<typename T>
        sparse_matrix_host<T> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);

        void destroy_sparse_matrix(sparse_matrix_device& device_matrix);

    }
}