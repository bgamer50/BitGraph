#include "bitgraph/strategy/BitGraphSelectionStrategy.h"
#include "bitgraph/step/BitGraphVStep.h"
#include "bitgraph/step/Fuzzy.h"

namespace bitgraph {
    gremlinxx::TraversalStrategy BitGraphSelectionStrategy = {
        gremlinxx::PROVIDER,
        "BitGraphSelectionStrategy",
            [](std::vector<std::shared_ptr<gremlinxx::TraversalStep>>& steps) {
            for(auto it = steps.begin(); it != steps.end(); ++it) {
                if((*it)->uid == BITGRAPH_V_STEP && it + 1 != steps.end()) {
                    BitGraphVStep* v_step = static_cast<BitGraphVStep*>((*it).get());
                    auto& next_step = (*(it+1));
                    if(next_step->uid == HAS_STEP) {
                        gremlinxx::HasStep* has_step = static_cast<gremlinxx::HasStep*>(next_step.get());
                        for(auto p : has_step->get_predicates()) {
                            v_step->add_predicate(
                                p.first,
                                p.second
                            );
                        }
                        
                        it = steps.erase(it+1) - 2;
                    } else if(next_step->uid == LIKE_STEP) {
                        gremlinxx::LikeStep* like_step = static_cast<gremlinxx::LikeStep*>(next_step.get());
                        v_step->add_fuzzy(bitgraph::fuzzy_t{
                            like_step->get_embeddings(),
                            like_step->get_embedding_name(),
                            like_step->get_embedding_stride(),
                            like_step->get_match_threshold(),
                            like_step->get_metric()
                        });

                        it = steps.erase(it+1) - 2;
                    }
                }
            }
        }
    };
}