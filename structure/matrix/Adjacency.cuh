#pragma once

#include "structure/memory/TypeErasure.cuh"
#include "structure/matrix/GPUSparseMatrix.cuh"
#include "util/cuda_utils.cuh"

#include <inttypes.h>
#include <cuda_runtime.h>

// Adjacency queries of sparse matrices
namespace bitgraph {
    namespace matrix {
        enum ADJ {VERTEX_TO_VERTEX=0, VERTEX_TO_EDGE=1, EDGE_TO_VERTEX=2, EDGE_TO_EDGE=3};

        /**
            M: (row_ptr, col_ptr) adjacency matrix
            T: intial traversers
            ps: prefix summed result array (indicates index of start for each traverser; last element is total size)
            O: The output array of traversers
            OO: The output origin array (which original traverser did the new traverser originate from)
            N: The original number of traversers
        **/
        __global__ void k_quadvv_get_adj(size_t* row_ptr, size_t* col_ptr, size_t* T, size_t* ps, size_t* O, size_t* OO, size_t N);

        /**
            Helper method for gpu_query_adjacency
            For each traverser, calculate the number of new traversers that will be produced.

            M: (row_ptr) adjacency matrix
            T: initial traversers
            R: result array; contains # of new traversers each traverser will generate
            N: # of initial traversers
        **/
        __global__ void k_quadvv_get_mem(size_t* row_ptr, size_t* T, size_t* R, size_t N);

        // (vertex id) -> (originating traverser)
        /**
            M: The sparse adjacency matrix on the GPU
            input_vertices: The traversers as literal objects (an array of of Vertex ids.)

            Returns a tuple of the output adjacent objects and the originating indices.
        **/
        std::tuple<bitgraph::memory::TypeErasedVector, bitgraph::memory::TypeErasedVector> gpu_query_adjacency_v_to_v(
            bitgraph::matrix::sparse_matrix_device& M,
            bitgraph::memory::TypeErasedVector& input_vertices,
            cudaStream_t stream=0
        );
    }
}