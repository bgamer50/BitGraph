#pragma once

#include "gremlinxx/gremlinxx.h"

#define BITGRAPH_V_STEP 0x11

namespace bitgraph {

    class BitGraphVStep : public gremlinxx::TraversalStep{
        private:
            std::vector<std::any> element_ids;

        public:
            BitGraphVStep(std::vector<std::any> element_ids);

            using gremlinxx::TraversalStep::apply;
            virtual void apply(gremlinxx::GraphTraversal* trv, gremlinxx::traversal::TraverserSet& traversers);

            using gremlinxx::TraversalStep::getInfo;
            virtual std::string getInfo() {
                return "BitGraphVStep()";
            }
    };

}