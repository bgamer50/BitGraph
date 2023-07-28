#include "bitgraph/structure/BitGraph.h"

namespace bitgraph {

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

        if(direction == gremlinxx::IN || direction == gremlinxx::BOTH) {
            // Here it is safe to transform in place
            this->matrix->to_csc();

            std::tie(result_vertices, result_origin, std::ignore, std::ignore) = this->matrix->query_adjacency(
                current_vertices,
                rel_types,
                true, // return_inner
                false, // return_values
                false // return_relations
            );

            if(result_vertices.get_mem_type() != this->traverser_storage) result_vertices = result_vertices.to(this->traverser_storage);
            if(result_origin.get_mem_type() != this->traverser_storage) result_origin = result_origin.to(this->traverser_storage);

        }

        if(direction == gremlinxx::OUT || direction == gremlinxx::BOTH) {
            this->matrix->to_csr();

            maelstrom::vector out_vertices;
            maelstrom::vector out_origin;

            std::tie(out_vertices, out_origin, std::ignore, std::ignore) = this->matrix->query_adjacency(
                current_vertices,
                rel_types,
                true, // return_inner
                false, // return_values
                false // return_relations
            );

            if(out_vertices.get_mem_type() != this->traverser_storage) out_vertices = out_vertices.to(this->traverser_storage);
            if(out_origin.get_mem_type() != this->traverser_storage) out_origin = result_vertices.to(this->traverser_storage);

            if(result_origin.empty()) {
                result_origin = std::move(out_origin);
                result_vertices = std::move(out_vertices);
            } else {
                result_origin.insert(result_origin.size(), out_origin);
                result_vertices.insert(result_vertices.size(), out_vertices);
            }
        }

        return std::make_pair(
            std::move(result_vertices),
            std::move(result_origin)            
        );
    }

}