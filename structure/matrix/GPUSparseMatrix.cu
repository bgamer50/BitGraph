#include "structure/matrix/GPUSparseMatrix.cuh"

#include "structure/memory/ThrustUtils.cuh"
#include "structure/memory/GPUDynamicCastingHelper.cuh"
#include "util/cuda_utils.cuh"

#include <sstream>
#include <inttypes.h>
#include <cuda_runtime.h>

namespace bitgraph {
    namespace matrix {
        /**
        * Converts a CSR sparse matrix on the host to a CSR matrix on the device. 
        **/
        template <typename T>
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<T>& host_matrix) {
            sparse_matrix_device device_matrix;
            device_matrix.nnz = host_matrix.nnz;
            device_matrix.num_rows = host_matrix.num_rows;
            device_matrix.num_cols = host_matrix.num_cols;
            device_matrix.value_dtype = gremlinxx::comparison::C_TYPEID<T>();

            cudaMalloc(&device_matrix.row_ptr, sizeof(size_t) * (host_matrix.num_rows + 1));
            cudaMemcpy(device_matrix.row_ptr, host_matrix.row_ptr.data(), sizeof(size_t) * (host_matrix.num_rows + 1), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            cudaCheckErrors("Copy row pointer to device");

            cudaMalloc(&device_matrix.col_ptr, sizeof(size_t) * host_matrix.nnz);
            cudaMemcpy(device_matrix.col_ptr, host_matrix.col_ptr.data(), sizeof(size_t) * host_matrix.nnz, cudaMemcpyDefault);
            cudaDeviceSynchronize();
            cudaCheckErrors("Copy col pointer to device");

            if(!host_matrix.values.empty()) {
                cudaMalloc(&device_matrix.values, sizeof(T) * host_matrix.nnz);
                cudaMemcpy(device_matrix.values, host_matrix.values.data(), sizeof(T) * host_matrix.nnz, cudaMemcpyDefault);
                cudaDeviceSynchronize();
                cudaCheckErrors("Copy values to device");
            }

            cudaDeviceSynchronize();
            cudaCheckErrors("Convert host sparse matrix to device sparse matrix");

            return device_matrix;
        }

        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<uint64_t>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<uint32_t>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<uint8_t>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<int64_t>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<int32_t>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<int8_t>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<float>& host_matrix);
        template
        sparse_matrix_device sparse_convert_host_to_device(sparse_matrix_host<double>& host_matrix);

        /**
        * Returns the left index of the COO matrix.
        * The right index is the already-existing col ptr.
        */
        size_t* csr_to_coo(sparse_matrix_device& device_matrix) {
            size_t* out_ix_left;
            cudaMalloc(&out_ix_left, sizeof(size_t) * device_matrix.nnz);
            cudaDeviceSynchronize();
            cudaCheckErrors("allocate out ix left");
            thrust::device_ptr<size_t> out_ix_left_tptr = thrust::device_pointer_cast(out_ix_left);

            // Run 32 streams
            const size_t n_streams = 32;
            std::vector<cudaStream_t> streams(32);
            for(size_t s = 0; s < n_streams; ++s) {
                cudaError_t error = cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
                if(error != cudaSuccess) throw std::runtime_error("Failed to create cuda stream");
            }

            for(size_t row = 0; row < device_matrix.num_rows; ++row) {
                auto& current_stream = streams[row % n_streams];

                auto start_end_ix = std::vector<size_t>(2);
                cudaMemcpyAsync(
                    start_end_ix.data(),
                    device_matrix.row_ptr + row,
                    sizeof(size_t) * 2,
                    cudaMemcpyDefault,
                    current_stream
                );
                cudaError_t error = cudaStreamSynchronize(current_stream);
                if(error != cudaSuccess) {
                    std::stringstream sx;
                    sx << "Failed to synchronize cuda stream while constructing coo: " << cudaGetErrorName(error) << std::endl << cudaGetErrorString(error);
                    throw std::runtime_error(sx.str());
                }

                size_t start_ix_incl = start_end_ix[0];
                size_t end_ix_excl = start_end_ix[1];

                // adapted from https://github.com/NVIDIA/thrust/blob/main/examples/cuda/explicit_cuda_stream.cu
                auto nosync_exec_policy = thrust::cuda::par_nosync.on(current_stream);

                auto filler = thrust::make_constant_iterator(row);
                thrust::copy(
                    nosync_exec_policy,
                    filler,
                    filler + (end_ix_excl - start_ix_incl),
                    out_ix_left_tptr + start_ix_incl
                );
            }

            // stop 32 streams
            for(size_t s = 0; s < n_streams; ++s) {
                cudaError_t error = cudaStreamSynchronize(streams[s]);
                if(error != cudaSuccess) {
                    std::stringstream sx;
                    sx << "Failed to synchronize cuda stream: " << cudaGetErrorName(error) << std::endl << cudaGetErrorString(error);
                    throw std::runtime_error(sx.str());
                }

                error = cudaStreamDestroy(streams[s]);
                if(error != cudaSuccess) {
                    std::stringstream sx;
                    sx << "Failed to destroy cuda stream: " << cudaGetErrorName(error) << std::endl << cudaGetErrorString(error);
                    throw std::runtime_error(sx.str());
                }
            }

            return out_ix_left;
        }

        template <typename T>
        void zipped_sort_coo(size_t* src, size_t* dst, void* values, size_t num_edges) {
            thrust::device_ptr<size_t> src_tptr = thrust::device_pointer_cast<size_t>(src);
            thrust::device_ptr<size_t> dst_tptr = thrust::device_pointer_cast<size_t>(dst);
            thrust::device_ptr<T> values_tptr = thrust::device_pointer_cast<T>(static_cast<T*>(values));
            auto zip_vals_begin = thrust::make_zip_iterator(
                thrust::make_tuple(dst_tptr, values_tptr)
            );

            thrust::stable_sort_by_key(
                thrust::device,
                src_tptr,
                src_tptr + num_edges,
                zip_vals_begin
            );
        }

        template
        void zipped_sort_coo<int8_t>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<int32_t>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<int64_t>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<uint8_t>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<uint32_t>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<uint64_t>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<float>(size_t* src, size_t* dst, void* values, size_t num_edges);
        template
        void zipped_sort_coo<double>(size_t* src, size_t* dst, void* values, size_t num_edges);

        void zipped_sort_coo_helper(size_t* src, size_t* dst, void* values, gremlinxx::comparison::C values_dtype, size_t num_edges) {
            switch(values_dtype) {
                case gremlinxx::comparison::C::UINT8:
                    return zipped_sort_coo<uint8_t>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::UINT32:
                    return zipped_sort_coo<uint32_t>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::UINT64:
                    return zipped_sort_coo<uint64_t>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::INT8:
                    return zipped_sort_coo<int8_t>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::INT32:
                    return zipped_sort_coo<int32_t>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::INT64:
                    return zipped_sort_coo<int64_t>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::FLOAT32:
                    return zipped_sort_coo<float>(src, dst, values, num_edges);
                case gremlinxx::comparison::C::FLOAT64:
                    return zipped_sort_coo<double>(src, dst, values, num_edges);
                default: {
                    std::stringstream sx;
                    sx << "Illegal matrix value type " << gremlinxx::comparison::C_to_string[values_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
        }

        /**
        * Transposes a CSR matrix on the device.
        **/
        sparse_matrix_device transpose_csr_matrix(sparse_matrix_device& device_matrix) {
            size_t* out_ix_left = csr_to_coo(device_matrix);
            
            size_t value_size = gremlinxx::comparison::C_size[device_matrix.value_dtype];
            size_t* shuffled_values = nullptr;
            if(device_matrix.values != nullptr) {
                cudaMalloc(&shuffled_values, value_size * device_matrix.nnz);
                cudaMemcpy(shuffled_values, device_matrix.values, value_size * device_matrix.nnz, cudaMemcpyDefault);
            }

            // At this point, (out_ix_left, device_matrix.col_ptr) makes up a COO
            // This COO needs to be reversed and transformed back to CSR
            // out_ix_left will become the new column pointer


            // have to create a copy of the col ptr since it needs to be resorted
            size_t* resorted_col_ptr;
            cudaMalloc(&resorted_col_ptr, sizeof(size_t) * device_matrix.nnz);

            thrust::copy(
                thrust::device,
                thrust::device_pointer_cast(device_matrix.col_ptr),
                thrust::device_pointer_cast(device_matrix.col_ptr) + device_matrix.nnz,
                thrust::device_pointer_cast(resorted_col_ptr)
            );

            if(device_matrix.values == nullptr) {
                // out_row_ptr is in order so doing a stable sort on col_ptr will ensure correct order.
                thrust::stable_sort_by_key(
                    thrust::device,
                    thrust::device_pointer_cast(resorted_col_ptr),
                    thrust::device_pointer_cast(resorted_col_ptr) + device_matrix.nnz,
                    thrust::device_pointer_cast<size_t>(out_ix_left)
                );
            }
            else {
                // values have to be resorted too if present
                // this is done in a helper function since dtype needs to be checked
                zipped_sort_coo_helper(
                    resorted_col_ptr,
                    out_ix_left,
                    shuffled_values,
                    device_matrix.value_dtype,
                    device_matrix.nnz
                );
            }

            size_t* row_keys;
            cudaMalloc(&row_keys, sizeof(size_t) * device_matrix.num_rows);

            size_t* row_indices;
            cudaMalloc(&row_indices, sizeof(size_t) * (device_matrix.num_rows + 1));

            auto new_end = thrust::reduce_by_key(
                thrust::device,
                thrust::device_pointer_cast(resorted_col_ptr),
                thrust::device_pointer_cast(resorted_col_ptr) + device_matrix.nnz,
                thrust::make_constant_iterator((size_t)1),
                thrust::device_pointer_cast<size_t>(row_keys),
                thrust::device_pointer_cast(row_indices) + 1
            );
            cudaFree(resorted_col_ptr);
            size_t num_non_empty_rows = new_end.second - thrust::device_pointer_cast(row_indices) - 1;

            size_t* out_row_ptr;
            cudaMalloc(&out_row_ptr, sizeof(size_t) * (device_matrix.num_rows + 1));
            cudaMemset(out_row_ptr, 0, sizeof(size_t) * (device_matrix.num_rows + 1)); // set all elements to 0 (necessary because some rows may be blanks)

            thrust::scatter(
                thrust::device,
                thrust::device_pointer_cast(row_indices) + 1,
                thrust::device_pointer_cast(row_indices) + num_non_empty_rows + 1,
                row_keys,
                thrust::device_pointer_cast(out_row_ptr) + 1
            );
            cudaFree(row_keys);
            cudaFree(row_indices);

            thrust::inclusive_scan(
                thrust::device,
                thrust::device_pointer_cast(out_row_ptr) + 1,
                thrust::device_pointer_cast(out_row_ptr) + device_matrix.num_rows + 1,
                thrust::device_pointer_cast(out_row_ptr) + 1
            );

            sparse_matrix_device output_matrix;
            output_matrix.num_rows = device_matrix.num_cols;
            output_matrix.num_cols = device_matrix.num_rows;
            output_matrix.nnz = device_matrix.nnz;
            output_matrix.row_ptr = out_row_ptr;
            output_matrix.col_ptr = out_ix_left;
            output_matrix.values = shuffled_values;
            
            return output_matrix;
        }

        /**
        * Converts a CSR matrix on the device to a CSR sparse matrix on the host. 
        **/
        template<typename T>
        sparse_matrix_host<T> sparse_convert_device_to_host(sparse_matrix_device& device_matrix) {
            sparse_matrix_host<T> host_matrix;

            host_matrix.num_rows = device_matrix.num_rows;
            host_matrix.num_cols = device_matrix.num_cols;
            host_matrix.nnz = device_matrix.nnz;

            host_matrix.row_ptr.resize(device_matrix.num_rows + 1);
            host_matrix.col_ptr.resize(device_matrix.nnz);

            thrust::copy(
                thrust::device_pointer_cast<size_t>(device_matrix.row_ptr),
                thrust::device_pointer_cast<size_t>(device_matrix.row_ptr) + device_matrix.num_rows + 1,
                host_matrix.row_ptr.begin()
            );

            thrust::copy(
                thrust::device_pointer_cast<size_t>(device_matrix.col_ptr),
                thrust::device_pointer_cast<size_t>(device_matrix.col_ptr) + device_matrix.nnz,
                host_matrix.col_ptr.begin()
            );

            if(device_matrix.values != nullptr) {
                auto desired_dtype = gremlinxx::comparison::C_TYPEID<T>();
                void* transmuted_values;
                if(desired_dtype == device_matrix.value_dtype) {
                    transmuted_values = device_matrix.values;
                }
                else {
                    // Have to transmute the values to the datatype desired by the user.
                    cudaMalloc(&transmuted_values, sizeof(T) * device_matrix.nnz);
                    bitgraph::memory::array_cast_outer_wrapper(
                        device_matrix.values,
                        transmuted_values,
                        device_matrix.nnz,
                        device_matrix.value_dtype,
                        desired_dtype
                    );
                }

                host_matrix.values.resize(device_matrix.nnz);
                thrust::copy(
                    thrust::device_pointer_cast<T>(static_cast<T*>(transmuted_values)),
                    thrust::device_pointer_cast<T>(static_cast<T*>(transmuted_values)) + device_matrix.nnz,
                    host_matrix.values.begin()
                );
            }

            return host_matrix;
        }

        template
        sparse_matrix_host<uint64_t> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<uint32_t> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<uint8_t> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<int64_t> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<int32_t> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<int8_t> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<float> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);
        template
        sparse_matrix_host<double> sparse_convert_device_to_host(sparse_matrix_device& device_matrix);


        void destroy_sparse_matrix(sparse_matrix_device& device_matrix) {
            cudaFree(device_matrix.values);
            cudaFree(device_matrix.row_ptr);
            cudaFree(device_matrix.col_ptr);
        }
    }
}