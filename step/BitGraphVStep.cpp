#include "bitgraph/structure/BitGraph.h"
#include "bitgraph/structure/Index.h"
#include "bitgraph/step/BitGraphVStep.h"

#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/filter.h"
#include "maelstrom/algorithms/remove_if.h"
#include "maelstrom/algorithms/intersection.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/topk.h"

#include "maelstrom/util/any_utils.h"

namespace bitgraph {
    bool BITGRAPH_VALIDITY_WARNING = false;

    BitGraphVStep::BitGraphVStep(maelstrom::vector element_ids)
    : gremlinxx::TraversalStep(gremlinxx::MAP, BITGRAPH_V_STEP) {
        this->element_ids = std::move(element_ids);
    }

    void BitGraphVStep::apply(gremlinxx::GraphTraversal* trv, gremlinxx::traversal::TraverserSet& traversers) {
        auto graph = static_cast<BitGraph*>(trv->getGraph());
        auto traverser_storage = graph->traverser_storage;

        maelstrom::vector query_vertices;

        size_t remaining_fuzzies = fuzzies.size();
        if(this->element_ids.empty()) {
            if(this->predicates.empty()) {
                if(this->fuzzies.empty()) {
                    query_vertices = maelstrom::arange(
                        traverser_storage,
                        this->limit ? std::min(graph->num_vertices(), *(this->limit)) : graph->num_vertices()
                    ).astype(graph->vertex_dtype);
                } else {
                    auto f = fuzzies.back();
                    auto emb = graph->vertex_embeddings.find(f.emb_name);
                    if(emb == graph->vertex_embeddings.end()) {
                        throw std::invalid_argument("No embedding with the given name exists!");
                    }

                    auto emb_index = graph->embedding_indices.find(f.emb_name);
                    bool has_index = (emb_index != graph->embedding_indices.end());

                    if(f.count && f.count.value() == 0) {
                        // do nothing
                    } else if(has_index && !f.match_threshold && f.count) {
                        query_vertices = std::move(
                            bitgraph::search_embedding_index_knn(
                                emb->second,
                                emb_index->second,
                                f.embeddings,
                                f.emb_stride,
                                f.count.value()
                            )
                        );
                        query_vertices = std::move(
                            query_vertices.astype(graph->vertex_dtype).to(graph->traverser_storage)
                        );
                    } else {
                        maelstrom::vector empty;
                        auto sim = maelstrom::similarity(
                            f.similarity_metric,
                            emb->second,
                            empty,
                            f.embeddings,
                            f.emb_stride
                        );

                        // Have to filter out -inf, nan, etc.
                        double threshold = f.match_threshold.value_or(std::numeric_limits<double>::min());
                        query_vertices = std::move(
                            maelstrom::filter(sim, maelstrom::GREATER_THAN_OR_EQUAL, threshold)
                        );

                        if(f.count) {
                            sim = std::move(
                                maelstrom::select(sim, query_vertices)
                            );
                            auto topk_ix = maelstrom::topk(sim, f.count.value());
                            query_vertices = std::move(maelstrom::select(query_vertices, topk_ix));
                        } 
                    }

                    query_vertices = query_vertices.astype(graph->vertex_dtype);
                    remaining_fuzzies -= 1;
                }
            } else {
                for(auto& p : this->predicates) {
                    // heuristic for whether to get all properties and do a set intersection instead of just a lookup
                    if(query_vertices.empty() || graph->get_vertex_property_num_entries(p.first) < query_vertices.size()) {
                        std::optional<maelstrom::vector> keys;
                        std::optional<maelstrom::vector> vals;
                        std::tie(keys, vals) = graph->view_vertex_property(p.first, true, p.second.operand.has_value());

                        if(p.second.operand.has_value()) {
                            auto ix = maelstrom::filter(*vals, p.second.comparison, p.second.operand);
                            *keys = std::move(maelstrom::select(*keys, ix));
                        }

                        maelstrom::sort(*keys);
                        if(query_vertices.empty()) {
                            query_vertices = std::move(*keys);
                        } else {
                            // intersection ensures the output is sorted
                            auto iix = maelstrom::intersection(query_vertices, *keys);
                            query_vertices = std::move(
                                maelstrom::select(query_vertices, iix)
                            );
                        }
                    } else {
                        // just do a lookup, it is probably faster and definitely more memory-efficient
                        maelstrom::vector values;
                        maelstrom::vector output_origin;
                        std::tie(values, output_origin) = graph->get_vertex_properties(p.first, query_vertices, p.second.operand.has_value());

                        if(p.second.operand.has_value()) {
                            auto ix = maelstrom::filter(values, maelstrom::EQUALS, p.second.operand);
                            output_origin = std::move(maelstrom::select(output_origin, ix));
                        }

                        query_vertices = std::move(
                            maelstrom::select(query_vertices, output_origin)
                        );
                    }

                    // immediately quit if all vertices were ruled out, otherwise will fill with invalid keys
                    if(query_vertices.empty()) break;
                }

                // Make sure the limit is applied at the end if present.
                // Resizing to the current size is a no-op in maelstrom.
                query_vertices.resize(std::min(query_vertices.size(), this->limit.value_or(query_vertices.size())));
            }
        } else {
            query_vertices = this->element_ids.astype(graph->vertex_dtype);
            
            // TODO properly check if vertices are valid.
            if(!BITGRAPH_VALIDITY_WARNING) {
                std::cerr << "warning: BitGraph currently does not check vertex validity" << std::endl;
                BITGRAPH_VALIDITY_WARNING = true;
            }
        }

        for(size_t k = remaining_fuzzies; k > 0; --k) {
            auto& f = fuzzies[k-1];
            maelstrom::vector empty;
            auto sim = maelstrom::similarity(
                f.similarity_metric,
                query_vertices,
                empty,
                f.embeddings,
                f.emb_stride
            );

            maelstrom::vector q;
            if(f.match_threshold) {
                auto q = maelstrom::filter(
                    sim,
                    maelstrom::GREATER_THAN_OR_EQUAL,
                    f.match_threshold.value()
                );
                query_vertices = std::move(
                    maelstrom::select(query_vertices, q)
                );
            }

            if(f.count) {
                if(f.match_threshold) {
                    sim = std::move(
                        maelstrom::select(sim, q)
                    );
                    auto topk_ix = maelstrom::topk(sim, f.count.value());
                    query_vertices = std::move(
                        maelstrom::select(query_vertices, topk_ix)
                    );
                } else {
                    auto q = maelstrom::topk(sim, f.count.value());
                    query_vertices = std::move(
                        maelstrom::select(query_vertices, q)
                    );
                }
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