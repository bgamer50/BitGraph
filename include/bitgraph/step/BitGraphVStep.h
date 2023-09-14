#pragma once

#include "gremlinxx/gremlinxx.h"

#include <optional>

#define BITGRAPH_V_STEP 0x11

namespace bitgraph {

    class BitGraphVStep : public gremlinxx::TraversalStep, public gremlinxx::LimitSupportingStep {
        private:
            std::vector<std::any> element_ids;
            std::optional<size_t> limit = {};

        public:
            BitGraphVStep(std::vector<std::any> element_ids);

            using gremlinxx::TraversalStep::apply;
            virtual void apply(gremlinxx::GraphTraversal* trv, gremlinxx::traversal::TraverserSet& traversers);

            using gremlinxx::LimitSupportingStep::set_limit;
            virtual void set_limit(size_t limit) { this->limit.emplace(limit); }

            using gremlinxx::LimitSupportingStep::get_limit;
            virtual std::optional<size_t> get_limit() { return this->limit; }

            using gremlinxx::TraversalStep::getInfo;
            virtual std::string getInfo() {
                return "BitGraphVStep()";
            }
    };

}