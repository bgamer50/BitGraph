#include "bitgraph/strategy/BitGraphStrategy.h"
#include "bitgraph/step/BitGraphVStep.h"

namespace bitgraph {

    gremlinxx::TraversalStrategy BitGraphStrategy = {
        gremlinxx::PROVIDER,
        "BitGraphStrategy",
            [](std::vector<std::shared_ptr<gremlinxx::TraversalStep>>& steps) {
            for(auto it = steps.begin(); it != steps.end(); ++it) {
                if((*it)->uid == V_STEP) {
                    *it = std::shared_ptr<gremlinxx::TraversalStep>(
                        new BitGraphVStep(
                            static_cast<gremlinxx::VStep*>(it->get())->get_element_ids()
                        )
                    );
                }
            }
        }
    };

}