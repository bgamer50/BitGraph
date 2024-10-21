#include "bitgraph/structure/BitGraph.h"

#include "maelstrom/algorithms/arange.h"
#include "maelstrom/util/any_utils.h"

namespace bitgraph {

    void BitGraph::to_canonical_csc() {
        if(this->matrix->get_format() == maelstrom::COO) {
            // We always enforce that the matrix is in canonical
            // format, so this is a safe operation.
            // In canonical coo format, there should be no values.

            auto raw_edge_dtype = maelstrom::dtype_from_prim_type(this->edge_dtype.prim_type);
            auto values = maelstrom::arange(
                this->structure_storage,
                maelstrom::safe_any_cast(this->matrix->num_nonzero(), raw_edge_dtype)
            ).astype(this->edge_dtype);

            this->matrix->set_values(std::move(values));
            this->matrix->to_csc();
            // The csc matrix now contains the correct permutation (as values)
            return;
        } else if(this->matrix->get_format() == maelstrom::CSR) {
            // So long as the permutation array is not disturbed, a
            // regular transformation is ok.
            // Once again, it is always safe to assume the matrix is
            // in canonical CSR format (which means values are present).
            this->matrix->to_csc();
            return;
        } else if(this->matrix->get_format() == maelstrom::CSC) {
            return; // do nothing, matrix is already canonical CSC
        }

        throw std::runtime_error("Invalid matrix format!");
    }

    void BitGraph::to_canonical_csr() {
        if(this->matrix->get_format() == maelstrom::COO) {
            auto raw_edge_dtype = maelstrom::dtype_from_prim_type(this->edge_dtype.prim_type);
            auto values = maelstrom::arange(
                this->structure_storage,
                maelstrom::safe_any_cast(this->matrix->num_nonzero(), raw_edge_dtype)
            ).astype(this->edge_dtype);
            
            this->matrix->set_values(std::move(values));
            this->matrix->to_csr();
            return;
        } else if(this->matrix->get_format() == maelstrom::CSC) {
            this->matrix->to_csr();
            return;
        } else if(this->matrix->get_format() == maelstrom::CSR) {
            return;
        }

        throw std::runtime_error("Invalid matrix format!");
    }

    void BitGraph::to_canonical_coo() {
        if(this->matrix->get_format() == maelstrom::COO) {
            return;
        } else if(this->matrix->get_format() == maelstrom::CSC || this->matrix->get_format() == maelstrom::CSR) {
            this->matrix->to_coo();
            this->matrix->sort_values();
            this->matrix->set_values(maelstrom::vector());
            return;
        }

        throw std::runtime_error("Invalid matrix format!");
    }

}