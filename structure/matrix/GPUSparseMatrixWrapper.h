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
 * Transposes a CSR matrix on the device.
**/
sparse_matrix_device_t transpose_csr_matrix(cusparseHandle_t handle, sparse_matrix_device_t& device_matrix) {
    sparse_matrix_device_t gpu_csc_matrix;
    gpu_csc_matrix.nnz = device_matrix.nnz;
    gpu_csc_matrix.num_rows = device_matrix.num_rows;
    gpu_csc_matrix.num_cols = device_matrix.num_cols;

    cudaMalloc((void **) &gpu_csc_matrix.row_ptr, sizeof(int32_t) * (gpu_csc_matrix.num_rows + 1));
    cudaMalloc((void **) &gpu_csc_matrix.col_ptr, sizeof(int32_t) * gpu_csc_matrix.nnz);
    cudaMalloc((void **) &gpu_csc_matrix.values, sizeof(float) * gpu_csc_matrix.nnz);

    // Convert the CSR matrix to CSC.  In this format, it is identical to the transpose of the CSR.
    size_t bufsize;
    cusparseStatus_t status_bufsize = cusparseCsr2cscEx2_bufferSize(
        handle,
        device_matrix.num_rows,
        device_matrix.num_cols,
        device_matrix.nnz,
        device_matrix.values,
        device_matrix.row_ptr,
        device_matrix.col_ptr,
        gpu_csc_matrix.values,
        gpu_csc_matrix.row_ptr,
        gpu_csc_matrix.col_ptr,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG2,
        &bufsize
    );
    cudaDeviceSynchronize();
    if(status_bufsize != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("error transposing device matrix (create buffer):\n" + std::string(cusparseGetErrorString(status_bufsize)));

    void* buffer;
    cudaMalloc(&buffer, bufsize);

    cusparseStatus_t status_convert = cusparseCsr2cscEx2(
        handle,
        device_matrix.num_rows,
        device_matrix.num_cols,
        device_matrix.nnz,
        device_matrix.values,
        device_matrix.row_ptr,
        device_matrix.col_ptr,
        gpu_csc_matrix.values,
        gpu_csc_matrix.row_ptr,
        gpu_csc_matrix.col_ptr,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG2,
        buffer
    );
    cudaDeviceSynchronize();
    if(status_convert != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("error transposing device matrix (convert csr to csc):\n" + std::string(cusparseGetErrorString(status_convert)));

    cudaFree(buffer);
    return gpu_csc_matrix;
}

/**
    Computes the matrix C = A + B, where A, B, and C are all m x n matrices.
    Adapted from the CUDA toolkit documentation (https://docs.nvidia.com/cuda/cusparse/index.html#csrgeam2)
**/
sparse_matrix_device_t add_csr_matrices(cusparseHandle_t handle, sparse_matrix_device_t& device_matrix_A, sparse_matrix_device_t& device_matrix_B) {
    const float alpha = 1;
    const float beta = 1;

    int baseC;
    
    size_t buffer_size_bytes;
    char* buffer = nullptr;

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    sparse_matrix_device_t device_matrix_C;
    cudaMalloc((void**)&device_matrix_C.row_ptr, sizeof(int)*(device_matrix_A.num_rows+1));
    int* nnzTotalDevHostPtr = &device_matrix_C.nnz;

    cusparseMatDescr_t descriptor;
    cusparseCreateMatDescr(&descriptor);
    
    /* prepare buffer */
    cusparseScsrgeam2_bufferSizeExt(handle, device_matrix_A.num_rows, device_matrix_A.num_cols,
        &alpha,
        descriptor, device_matrix_A.nnz,
        device_matrix_A.values, device_matrix_A.row_ptr, device_matrix_A.col_ptr,
        &beta,
        descriptor, device_matrix_B.nnz,
        device_matrix_B.values, device_matrix_B.row_ptr, device_matrix_B.col_ptr,
        descriptor,
        device_matrix_C.values, device_matrix_C.row_ptr, device_matrix_C.col_ptr,
        &buffer_size_bytes
    );
    cudaMalloc((void**)&buffer, sizeof(char) * buffer_size_bytes);

    cusparseXcsrgeam2Nnz(handle, device_matrix_A.num_rows, device_matrix_A.num_cols,
        descriptor, device_matrix_A.nnz, device_matrix_A.row_ptr, device_matrix_A.col_ptr,
        descriptor, device_matrix_B.nnz, device_matrix_B.row_ptr, device_matrix_B.col_ptr,
        descriptor, device_matrix_C.row_ptr, nnzTotalDevHostPtr,
        buffer
    );

    if (nullptr != nnzTotalDevHostPtr) {
        device_matrix_C.nnz = *nnzTotalDevHostPtr;
    } else {
        cudaMemcpy(&device_matrix_C.nnz, device_matrix_C.row_ptr + device_matrix_A.num_rows, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, device_matrix_C.row_ptr, sizeof(int), cudaMemcpyDeviceToHost);
        device_matrix_C.nnz -= baseC;
    }

    cudaMalloc((void**)&device_matrix_C.col_ptr, sizeof(int) * device_matrix_C.nnz);
    cudaMalloc((void**)&device_matrix_C.values, sizeof(float) * device_matrix_C.nnz);

    cusparseScsrgeam2(handle, device_matrix_A.num_rows, device_matrix_A.num_cols,
        &alpha,
        descriptor, device_matrix_A.nnz,
        device_matrix_A.values, device_matrix_A.row_ptr, device_matrix_A.col_ptr,
        &beta,
        descriptor, device_matrix_B.nnz,
        device_matrix_B.values, device_matrix_B.row_ptr, device_matrix_B.col_ptr,
        descriptor,
        device_matrix_C.values, device_matrix_C.row_ptr, device_matrix_C.col_ptr,
        buffer
    );

    return device_matrix_C;
}

/**
 * Converts a wrapped cusparse CSR matrix on the device to a CSR sparse matrix on the host. 
 **/
sparse_matrix_t sparse_convert_device_to_host(cusparseHandle_t handle, sparse_matrix_device_t& device_matrix);

void destroy_sparse_matrix(cusparseHandle_t handle, sparse_matrix_device_t& device_matrix) {
    //cusparseDestroySpMat(device_matrix.descriptor);
    cudaFree(device_matrix.values);
    cudaFree(device_matrix.row_ptr);
    cudaFree(device_matrix.col_ptr);
}

#endif