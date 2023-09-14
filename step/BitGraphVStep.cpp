#include "bitgraph/structure/BitGraph.h"
#include "bitgraph/step/BitGraphVStep.h"

#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/increment.h"

#include "maelstrom/util/any_utils.h"

namespace bitgraph {
    bool BITGRAPH_VALIDITY_WARNING = false;

    BitGraphVStep::BitGraphVStep(std::vector<std::any> element_ids)
    : gremlinxx::TraversalStep(gremlinxx::MAP, BITGRAPH_V_STEP) {
        this->element_ids = element_ids;
    }

    void BitGraphVStep::apply(gremlinxx::GraphTraversal* trv, gremlinxx::traversal::TraverserSet& traversers) {
        auto graph = static_cast<BitGraph*>(trv->getGraph());
        auto traverser_storage = graph->traverser_storage;

        maelstrom::vector query_vertices;

        if(this->element_ids.empty()) {
            query_vertices = maelstrom::arange(
                traverser_storage,
                this->limit ? std::min(graph->num_vertices(), *(this->limit)) : graph->num_vertices()
            ).astype(graph->vertex_dtype);
        } else {
            query_vertices = maelstrom::make_vector_from_anys(graph->traverser_storage, this->element_ids).astype(graph->vertex_dtype);
            
            // TODO properly check if vertices are valid.
            if(!BITGRAPH_VALIDITY_WARNING) {
                std::cerr << "warning: BitGraph currently does not check vertex validity" << std::endl;
                BITGRAPH_VALIDITY_WARNING = true;
            }
        }

        size_t num_traversers = traversers.size();

        // Simulate looping over each traverser
        if(traversers.empty()) {
            traversers.reinitialize(
                query_vertices,
                {},
                {}
            );
        } else if(num_traversers == 1) {
            traversers.advance(
                [&query_vertices, traverser_storage](auto& traverser_data, auto& traverser_side_effects, auto& traverser_path_info){
                    maelstrom::vector origin(
                        traverser_storage,
                        maelstrom::uint64,
                        query_vertices.size()        
                    );
                    maelstrom::set(origin, static_cast<size_t>(0));

                    return std::make_pair(
                        std::move(query_vertices),
                        std::move(origin)
                    );
                }
            );
        } else {
            traversers.advance(
                [&query_vertices, traverser_storage](auto& traverser_data, auto& traverser_side_effects, auto& traverser_path_info){
                    maelstrom::vector result_vertices(
                        traverser_storage,
                        query_vertices.get_dtype()
                    );
                    result_vertices.reserve(query_vertices.size() * traverser_data.size());

                    maelstrom::vector query_origin(
                        traverser_storage,
                        maelstrom::uint64,
                        query_vertices.size()
                    );
                    maelstrom::set(query_origin, static_cast<size_t>(0));

                    maelstrom::vector result_origin(
                        traverser_storage,
                        maelstrom::uint64
                    );
                    result_origin.reserve(query_vertices.size() * traverser_data.size());

                    result_vertices.insert(query_vertices);
                    result_origin.insert(query_origin);

                    for(size_t k = 1; k < traverser_data.size(); ++k) {
                        maelstrom::increment(query_origin);
                        result_vertices.insert(query_vertices);
                        result_origin.insert(query_origin);
                    }

                    return std::make_pair(
                        std::move(result_vertices),
                        std::move(result_origin)
                    );
                }
            );
        }
    }
}