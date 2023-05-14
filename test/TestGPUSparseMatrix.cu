#include <assert.h>

#include "test/TestUtils.hpp"

#include <cuda_runtime.h>

#include "structure/matrix/CPUSparseMatrix.h"
#include "structure/matrix/GPUSparseMatrix.cuh"

using namespace bitgraph::matrix;
using namespace bitgraph::test;

void test_transpose_device_simple();
void test_transpose_device_complex();
void test_csr_to_coo_device();

int main(int argc, char* argv[]) {
    try {
        test_csr_to_coo_device();
        test_transpose_device_simple();
        test_transpose_device_complex();

    } catch(const std::exception& err) {
        std::cout << err.what() << "\n";
        return -1;
    }
}

void test_transpose_device_simple() {
    /*
        1 0 1 0
        0 1 0 0
        0 0 1 0
        0 1 0 0
    */
    std::vector<size_t> col_ptr = {0, 2, 1, 2, 1};
    std::vector<size_t> row_ptr = {0, 2, 3, 4, 5};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    bitgraph::matrix::sparse_matrix_host<float> cpu_csr_matrix;
    cpu_csr_matrix.num_rows = 4;
    cpu_csr_matrix.num_cols = 4;
    cpu_csr_matrix.nnz = 5;
    cpu_csr_matrix.row_ptr = row_ptr;
    cpu_csr_matrix.col_ptr = col_ptr;
    cpu_csr_matrix.values = values;

    sparse_matrix_device gpu_csr_matrix = sparse_convert_host_to_device<float>(cpu_csr_matrix);
    sparse_matrix_device gpu_transposed_csr_matrix = transpose_csr_matrix(gpu_csr_matrix);
    sparse_matrix_host<float> cpu_transposed_csr_matrix = sparse_convert_device_to_host<float>(gpu_transposed_csr_matrix);
    std::cout << "transpose done! " << std::endl;

    for(size_t k = 0; k < cpu_transposed_csr_matrix.row_ptr.size(); ++k) std::cout << cpu_transposed_csr_matrix.row_ptr[k] << " ";
    std::cout << std::endl;
    std::vector<size_t> correct_row_ptr = {0, 1, 3, 5, 5};
    assert_vector_equals(correct_row_ptr, cpu_transposed_csr_matrix.row_ptr);
    

    for(size_t k = 0; k < cpu_transposed_csr_matrix.col_ptr.size(); ++k) std::cout << cpu_transposed_csr_matrix.col_ptr[k] << " ";
    std::cout << std::endl;
    std::vector<size_t> correct_col_ptr = {0, 1, 3, 0, 2};
    assert_vector_equals(correct_col_ptr, cpu_transposed_csr_matrix.col_ptr);

    for(size_t k = 0; k < cpu_transposed_csr_matrix.values.size(); ++k) std::cout << cpu_transposed_csr_matrix.values[k] << " ";
    std::cout << std::endl;
    std::vector<float> correct_values = {1, 3, 5, 2, 4};
    assert_vector_equals(correct_values, cpu_transposed_csr_matrix.values);
}

void test_transpose_device_complex() {
    std::vector<size_t> row_ptr = {0, 1, 2, 3, 4, 5, 5, 6};
    std::vector<size_t> col_ptr = {1, 2, 4, 2, 5, 5};
    std::vector<float> values = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};

    bitgraph::matrix::sparse_matrix_host<float> cpu_csr_matrix;
    cpu_csr_matrix.num_rows = 7;
    cpu_csr_matrix.num_cols = 7;
    cpu_csr_matrix.nnz = 6;
    cpu_csr_matrix.row_ptr = row_ptr;
    cpu_csr_matrix.col_ptr = col_ptr;
    cpu_csr_matrix.values = values;

    sparse_matrix_device gpu_csr_matrix = sparse_convert_host_to_device<float>(cpu_csr_matrix);
    sparse_matrix_device gpu_transposed_csr_matrix = transpose_csr_matrix(gpu_csr_matrix);
    sparse_matrix_host<float> cpu_transposed_csr_matrix = sparse_convert_device_to_host<float>(gpu_transposed_csr_matrix);
    std::cout << "transpose done! " << std::endl;

    for(size_t k = 0; k < cpu_transposed_csr_matrix.row_ptr.size(); ++k) std::cout << cpu_transposed_csr_matrix.row_ptr[k] << " ";
    std::cout << std::endl;
    std::vector<size_t> correct_row_ptr = {0, 0, 1, 3, 3, 4, 6, 6};
    assert_vector_equals(correct_row_ptr, cpu_transposed_csr_matrix.row_ptr);

    for(size_t k = 0; k < cpu_transposed_csr_matrix.col_ptr.size(); ++k) std::cout << cpu_transposed_csr_matrix.col_ptr[k] << " ";
    std::cout << std::endl;
    std::vector<size_t> correct_col_ptr = {0, 1, 3, 2, 4, 6};
    assert_vector_equals(correct_col_ptr, cpu_transposed_csr_matrix.col_ptr);

    for(size_t k = 0; k < cpu_transposed_csr_matrix.values.size(); ++k) std::cout << cpu_transposed_csr_matrix.values[k] << " ";
    std::cout << std::endl;
    std::vector<float> correct_values = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    assert_vector_equals(correct_values, cpu_transposed_csr_matrix.values);
}

void test_csr_to_coo_device() {
        /*
            1 0 1 0
            0 1 0 0
            0 0 1 0
            0 1 0 0
        */
        std::vector<size_t> col_ptr = {0, 2, 1, 2, 1};
        std::vector<size_t> row_ptr = {0, 2, 3, 4, 5};
        std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        bitgraph::matrix::sparse_matrix_host<float> cpu_csr_matrix;
        cpu_csr_matrix.num_rows = 4;
        cpu_csr_matrix.num_cols = 4;
        cpu_csr_matrix.nnz = 5;
        cpu_csr_matrix.row_ptr = row_ptr;
        cpu_csr_matrix.col_ptr = col_ptr;
        cpu_csr_matrix.values = values;

        sparse_matrix_device gpu_csr_matrix = sparse_convert_host_to_device<float>(cpu_csr_matrix);
        size_t* gpu_coo_left = csr_to_coo(gpu_csr_matrix);

        std::vector<size_t> cpu_coo_left(cpu_csr_matrix.nnz);
        cudaMemcpy(cpu_coo_left.data(), gpu_coo_left, sizeof(size_t) * cpu_csr_matrix.nnz, cudaMemcpyDefault);

        std::vector<size_t> correct_coo_left = {0, 0, 1, 2, 3};

        for(size_t k = 0; k < cpu_csr_matrix.nnz; ++k) assert( correct_coo_left[k] == cpu_coo_left[k] );
}