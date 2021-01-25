#ifndef GPU_SPARSE_MATRIX_WRAPPER_H
#define GPU_SPARSE_MATRIX_WRAPPER_H

#include <vector>
#include <inttypes.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "structure/matrix/CPUSparseMatrix.h"

typedef struct sparse_matrix_device {
    int32_t nnz; // number of nonzero elements
    float* values; // element values, device pointer
    int32_t* row_ptr; // row device pointer
    int32_t* col_ptr; // column device pointer

    int32_t num_rows;
    int32_t num_cols;

    cusparseSpMatDescr_t descriptor; // Sparse matrix descriptor, represents device data
} sparse_matrix_device_t;

/**
 * Converts a CSR sparse matrix on the host to a wrapped cusparse CSR matrix on the device. 
 **/
sparse_matrix_device_t sparse_convert_host_to_device(cusparseHandle_t handle, sparse_matrix_t& host_matrix) {
    sparse_matrix_device_t device_matrix;
    device_matrix.nnz = host_matrix.nnz;
    device_matrix.num_rows = host_matrix.num_rows;
    device_matrix.num_cols = host_matrix.num_cols;

    cudaMalloc((void **) &device_matrix.row_ptr, sizeof(int32_t) * host_matrix.row_ptr.size());
    cudaMemcpy(device_matrix.row_ptr, host_matrix.row_ptr.data(), sizeof(int32_t) * host_matrix.row_ptr.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &device_matrix.col_ptr, sizeof(int32_t) * host_matrix.nnz);
    cudaMemcpy(device_matrix.col_ptr, host_matrix.col_ptr.data(), sizeof(int32_t) * host_matrix.nnz, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &device_matrix.values, sizeof(float) * host_matrix.nnz);
    cudaMemcpy(device_matrix.values, host_matrix.values.data(), sizeof(float) * host_matrix.nnz, cudaMemcpyHostToDevice);

    cusparseStatus_t status = cusparseCreateCsr(
                                                &device_matrix.descriptor, 
                                                device_matrix.num_rows,
                                                device_matrix.num_cols,
                                                device_matrix.nnz, 
                                                device_matrix.row_ptr, 
                                                device_matrix.col_ptr, 
                                                device_matrix.values, 
                                                CUSPARSE_INDEX_32I,
                                                CUSPARSE_INDEX_32I, 
                                                CUSPARSE_INDEX_BASE_ZERO, 
                                                CUDA_R_32F
                                                );

    if(status != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("error converting host matrix to device matrix:\n" + std::string(cusparseGetErrorString(status)));

    return device_matrix;
}

/**
 * Converts a wrapped cusparse CSR matrix on the device to a CSR sparse matrix on the host. 
 **/
sparse_matrix_t sparse_convert_device_to_host(cusparseHandle_t handle, sparse_matrix_device_t& device_matrix);
#endif