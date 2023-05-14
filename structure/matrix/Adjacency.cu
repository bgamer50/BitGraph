#include "structure/matrix/Adjacency.cuh"

#include "structure/memory/ThrustUtils.cuh"
#include "structure/memory/TypeErasure.cuh"
#include "structure/matrix/GPUSparseMatrix.cuh"
#include "util/cuda_utils.cuh"

namespace bitgraph {
    namespace matrix {
        /**
            M: (row_ptr, col_ptr) adjacency matrix
            T: intial traversers
            ps: prefix summed result array (indicates index of start for each traverser; last element is total size)
            O: The output array of traversers
            OO: The output origin array (which original traverser did the new traverser originate from)
            N: The original number of traversers
        **/
        __global__ void k_quadvv_get_adj(size_t* row_ptr, size_t* col_ptr, size_t* T, size_t* ps, size_t* O, size_t* OO, size_t N) {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for(size_t i = index; i < N; i += stride) {
                size_t vertex = T[i];
                size_t output_index = i==0 ? 0 : ps[i - 1];

                size_t start = row_ptr[vertex];
                size_t end = row_ptr[vertex + 1];
                
                for(size_t j = start; j < end; ++j) {
                    O[output_index] = col_ptr[j];
                    OO[output_index] = i;
                    ++output_index;
                }
            }
        }

        /**
            Helper method for gpu_query_adjacency
            For each traverser, calculate the number of new traversers that will be produced.

            M: (row_ptr) adjacency matrix
            T: initial traversers
            R: result array; contains # of new traversers each traverser will generate
            N: # of initial traversers
        **/
        __global__ void k_quadvv_get_mem(size_t* row_ptr, size_t* T, size_t* R, size_t N) {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;
            for (size_t i = index; i < N; i += stride) {
                size_t vertex = T[i];
                size_t start = row_ptr[vertex];
                size_t end = row_ptr[vertex + 1];
                
                size_t v_out_count = end - start;
                R[i] = v_out_count;
            }
        }

        // (vertex id) -> (originating traverser)
        /**
            M: The sparse adjacency matrix on the GPU
            input_vertices: The traversers as literal objects (an array of of Vertex ids.)

            Returns a tuple of the output adjacent objects and the originating indices.
        **/
        std::tuple<bitgraph::memory::TypeErasedVector, bitgraph::memory::TypeErasedVector> gpu_query_adjacency_v_to_v(
            bitgraph::matrix::sparse_matrix_device& M,
            bitgraph::memory::TypeErasedVector& input_vertices,
            cudaStream_t stream
        ) {
            size_t* result;
            size_t* output;
            size_t* output_origin;
            size_t N = input_vertices.size();
            
            cudaMallocAsync(&result, sizeof(size_t) * N, stream);
            cudaStreamSynchronize(stream);
            cudaCheckErrors("allocate result");

            k_quadvv_get_mem<<<NUM_BLOCKS(N), BLOCK_SIZE, 0, stream>>>(M.row_ptr, static_cast<size_t*>(input_vertices.data()), result, N);
            cudaStreamSynchronize(stream);
            cudaCheckErrors("k_quadvv_get_mem");

            // prefix-sum the result array
            auto exec_policy = thrust::cuda::par.on(stream);
            auto counter = thrust::make_counting_iterator(static_cast<size_t>(0));
            thrust::inclusive_scan(
                exec_policy,
                thrust::device_pointer_cast<size_t>(result),
                thrust::device_pointer_cast<size_t>(result) + N,
                thrust::device_pointer_cast<size_t>(result)
            );
            cudaStreamSynchronize(stream);
            cudaCheckErrors("transform inclusive scan");

            // CPU needs to know ps information anyways, so we copy it and cudaMalloc the sum
            size_t N_prime; // = result[N-1]; # of output traversers
            cudaMemcpyAsync(&N_prime, &result[N-1], sizeof(size_t) * 1, cudaMemcpyDefault, stream);
            cudaStreamSynchronize(stream);
            cudaCheckErrors("copy result to host");
            
            cudaMallocAsync(&output, sizeof(size_t) * N_prime, stream);
            cudaMallocAsync(&output_origin, sizeof(size_t) * N_prime, stream);
            cudaStreamSynchronize(stream);
            cudaCheckErrors("allocate output and output origin");

            // Then we run a kernel that actually spits out the column #s (a.k.a. adjacent vertices in the out-direction, or in-direction if this matrix has been transposed)
            k_quadvv_get_adj<<<NUM_BLOCKS(N), BLOCK_SIZE, 0, stream>>>(M.row_ptr, M.col_ptr, static_cast<size_t*>(input_vertices.data()), result, output, output_origin, N);
            cudaStreamSynchronize(stream);
            cudaCheckErrors("k_quadvv_get_adj");
            
            bitgraph::memory::TypeErasedVector output_vec(
                bitgraph::memory::memory_type::DEVICE,
                gremlinxx::comparison::C::UINT64,
                output,
                N_prime,
                true
            );
            output_vec.own();

            bitgraph::memory::TypeErasedVector output_origin_vec(
                bitgraph::memory::memory_type::DEVICE,
                gremlinxx::comparison::C::UINT64,
                output_origin,
                N_prime,
                true
            );
            output_origin_vec.own();

            return std::make_tuple(
                std::move(output_vec),
                std::move(output_origin_vec)
            );
        }
    }
}