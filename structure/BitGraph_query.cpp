#include "bitgraph/structure/BitGraph.h"

#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/reduce_by_key.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/util/any_utils.h"

namespace bitgraph {

    std::pair<maelstrom::vector, maelstrom::vector> simple_v_adjacency_query(maelstrom::sparse_matrix& sparse_matrix, maelstrom::vector& current_vertices, maelstrom::vector& rel_types) {
        maelstrom::vector output_vertices;
        maelstrom::vector output_origin;
        std::tie(output_origin, output_vertices, std::ignore, std::ignore) = sparse_matrix.query_adjacency(
            current_vertices,
            rel_types,
            true, // return_inner
            false, // return_values
            false // return_relations
        );

        return std::make_pair(
            std::move(output_vertices),
            std::move(output_origin)
        );
    }

    std::pair<maelstrom::vector, maelstrom::vector> BitGraph::degree(maelstrom::vector& current_vertices, std::vector<std::string>& labels, gremlinxx::Direction direction) {
        std::vector<std::any> any_rels;
        any_rels.reserve(labels.size());
        any_rels.insert(any_rels.end(), labels.begin(), labels.end());
        maelstrom::vector rel_types = maelstrom::make_vector_from_anys(
            this->structure_storage,
            this->edge_label_index.get_dtype(),
            any_rels
        );

        if(this->matrix->get_format() == maelstrom::COO) {
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                this->to_canonical_csr();
            } else {
                this->to_canonical_csc();
            }
        }

        maelstrom::vector ix;
        maelstrom::vector counts;
        if(this->matrix->get_format() == maelstrom::CSR) {
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                std::tie(ix, counts) = this->matrix->nnz_i(current_vertices, rel_types);
            }
            if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
                this->to_canonical_csc();
                maelstrom::vector ix_r;
                maelstrom::vector counts_r;
                std::tie(ix_r, counts_r) = this->matrix->nnz_i(current_vertices, rel_types);
                if(ix.empty()) {
                    ix = std::move(ix_r);
                    counts = std::move(counts_r);
                } else {
                    ix.insert(ix_r);
                    counts.insert(counts_r);
                    ix_r.clear();
                    counts_r.clear();
                    
                    auto six = maelstrom::sort(ix);
                    counts = maelstrom::select(counts, six);
                    six.clear();
                    
                    std::tie(counts, ix) = maelstrom::reduce_by_key(ix, counts, maelstrom::SUM, true);
                }
            }
        } else if(this->matrix->get_format() == maelstrom::CSC) {
            if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
                std::tie(ix, counts) = this->matrix->nnz_i(current_vertices, rel_types);
            }
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                this->to_canonical_csr();
                maelstrom::vector ix_r;
                maelstrom::vector counts_r;
                std::tie(ix_r, counts_r) = this->matrix->nnz_i(current_vertices, rel_types);
                if(ix.empty()) {
                    ix = std::move(ix_r);
                    counts = std::move(counts_r);
                } else {
                    ix.insert(ix_r);
                    counts.insert(counts_r);
                    ix_r.clear();
                    counts_r.clear();
                    
                    auto six = maelstrom::sort(ix);
                    counts = std::move(maelstrom::select(counts, six));
                    six.clear();
                    
                    std::tie(counts, ix) = maelstrom::reduce_by_key(ix, counts, maelstrom::SUM, true);
                }
            }
        }

        return std::make_pair(
            std::move(counts),
            std::move(ix)
        );
    }

    std::pair<maelstrom::vector, maelstrom::vector> BitGraph::V(maelstrom::vector& current_vertices, std::vector<std::string>& labels, gremlinxx::Direction direction) {
        std::vector<std::any> any_rels;
        any_rels.reserve(labels.size());
        any_rels.insert(any_rels.end(), labels.begin(), labels.end());
        maelstrom::vector rel_types = maelstrom::make_vector_from_anys(
            this->structure_storage,
            this->edge_label_index.get_dtype(),
            any_rels
        );

        maelstrom::vector result_vertices;
        maelstrom::vector result_origin;

        if(this->matrix->get_format() == maelstrom::COO) {
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                this->to_canonical_csr();
            } else {
                this->to_canonical_csc();
            }
        }

        if(this->matrix->get_format() == maelstrom::CSR) {
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                std::tie(result_vertices, result_origin) = simple_v_adjacency_query(
                    *this->matrix,
                    current_vertices,
                    rel_types
                );
            }
            if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
                this->to_canonical_csc();
                maelstrom::vector next_vertices;
                maelstrom::vector next_origin;
                std::tie(next_vertices, next_origin) = simple_v_adjacency_query(
                    *this->matrix,
                    current_vertices,
                    rel_types
                );
                if(result_vertices.empty()) {
                    result_vertices = std::move(next_vertices);
                    result_origin = std::move(next_origin);
                } else {
                    result_vertices.insert(result_vertices.size(), next_vertices);
                    next_vertices.clear();
                    result_origin.insert(result_origin.size(), next_origin);
                    next_origin.clear();
                }
            }
        } else if(this->matrix->get_format() == maelstrom::CSC) {
            if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
                std::tie(result_vertices, result_origin) = simple_v_adjacency_query(
                    *this->matrix,
                    current_vertices,
                    rel_types
                );
            }
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                this->to_canonical_csr();
                maelstrom::vector next_vertices;
                maelstrom::vector next_origin;
                std::tie(next_vertices, next_origin) = simple_v_adjacency_query(
                    *this->matrix,
                    current_vertices,
                    rel_types
                );
                if(result_vertices.empty()) {
                    result_vertices = std::move(next_vertices);
                    result_origin = std::move(next_origin);
                } else {
                    result_vertices.insert(result_vertices.size(), next_vertices);
                    next_vertices.clear();
                    result_origin.insert(result_origin.size(), next_origin);
                    next_origin.clear();
                }
            }
        }

        if(result_vertices.get_mem_type() != this->traverser_storage) result_vertices = result_vertices.to(this->traverser_storage);
        if(result_origin.get_mem_type() != this->traverser_storage) result_origin = result_origin.to(this->traverser_storage);

        return std::make_pair(
            std::move(result_vertices),
            std::move(result_origin)            
        );
    }

    std::pair<maelstrom::vector, maelstrom::vector> simple_e_adjacency_query(maelstrom::sparse_matrix& sparse_matrix, maelstrom::vector& current_vertices, maelstrom::vector& rel_types) {
        maelstrom::vector original_indices;
        maelstrom::vector edge_ids;
        std::tie(original_indices, std::ignore, edge_ids, std::ignore) = sparse_matrix.query_adjacency(
            current_vertices,
            rel_types,
            false, // return_inner
            true, // return_values,
            false, // return_relations,
            false // return_1d_index_as_values
        );

        return std::make_pair(
            std::move(edge_ids),
            std::move(original_indices)
        );
    }

	std::pair<maelstrom::vector, maelstrom::vector> BitGraph::E(maelstrom::vector& current_vertices, std::vector<std::string>& labels, gremlinxx::Direction direction) {
        std::vector<std::any> any_rels;
        any_rels.reserve(labels.size());
        any_rels.insert(any_rels.end(), labels.begin(), labels.end());
        maelstrom::vector rel_types = maelstrom::make_vector_from_anys(
            this->structure_storage,
            this->edge_label_index.get_dtype(),
            any_rels
        );

        maelstrom::vector result_edges;
        maelstrom::vector result_origin;

        if(this->matrix->get_format() == maelstrom::COO) {
            if(direction == gremlinxx::OUT || direction == gremlinxx::IN) {
                this->to_canonical_csr();
            } else {
                this->to_canonical_csc();
            }
        }

        if(this->matrix->get_format() == maelstrom::CSR) {
            if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
                std::tie(result_edges, result_origin) = simple_e_adjacency_query(
                    *this->matrix,
                    current_vertices,
                    rel_types
                );
            }
            if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
                maelstrom::vector current_edges;
                maelstrom::vector current_origin;
                std::tie(current_edges, current_origin) = simple_e_adjacency_query(
                    *this->matrix,
                    current_vertices,
                    rel_types
                );

                if(result_edges.empty()) {
                    result_edges = std::move(current_edges);
                    result_origin = std::move(current_origin);
                } else {
                    result_edges.insert(result_edges.size(), current_edges);
                    current_edges.clear();

                    result_origin.insert(result_origin.size(), current_origin);
                    current_origin.clear();
                }
            }
        }

        if(result_edges.get_mem_type() != this->traverser_storage) result_edges = result_edges.to(traverser_storage);
        if(result_origin.get_mem_type() != this->traverser_storage) result_origin = result_origin.to(traverser_storage);

        return std::make_pair(
            std::move(result_edges),
            std::move(result_origin)
        );
    }

    std::pair<maelstrom::vector, maelstrom::vector> BitGraph::toV(maelstrom::vector& current_edges, gremlinxx::Direction direction) {
        // Have to convert to COO to properly interpret edge ids.
        this->to_canonical_coo();

        maelstrom::vector result_vertices;
        maelstrom::vector result_origin;

        if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
            result_vertices = this->matrix->get_rows_1d(current_edges);
            result_origin = maelstrom::arange(
                this->traverser_storage,
                current_edges.size()
            );
            for(size_t k = 0; k < 10; ++k) std::cout << std::any_cast<gremlinxx::Vertex>(result_vertices.get(k)).id << std::endl;
        }

        if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
            maelstrom::vector current_vertices = this->matrix->get_cols_1d(current_edges);
            auto current_origin = maelstrom::arange(
                this->traverser_storage,
                current_edges.size()
            );

            if(result_vertices.empty()) {
                result_vertices = std::move(current_vertices);
                result_origin = std::move(current_origin);
            } else {
                result_vertices.insert(current_vertices);
                current_vertices.clear();
                result_origin.insert(current_origin);
                result_origin.clear();
            }
        }

        if(result_vertices.get_mem_type() != this->traverser_storage) result_vertices = result_vertices.to(traverser_storage);
        if(result_origin.get_mem_type() != this->traverser_storage) result_origin = result_origin.to(traverser_storage);

        return std::make_pair(
            std::move(result_vertices),
            std::move(result_origin)
        );
    }


}