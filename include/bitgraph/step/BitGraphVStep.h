#pragma once

#include "gremlinxx/gremlinxx.h"
#include "bitgraph/step/Fuzzy.h"

#include <optional>

#define BITGRAPH_V_STEP 0x11

namespace bitgraph {

    class BitGraphVStep : public gremlinxx::TraversalStep, public gremlinxx::LimitSupportingStep {
        private:
            maelstrom::vector element_ids;
            std::optional<size_t> limit = {};
            std::vector<std::pair<std::string, gremlinxx::P>> predicates;
            std::vector<fuzzy_t> fuzzies;

        public:
            BitGraphVStep(maelstrom::vector element_ids);

            using gremlinxx::TraversalStep::apply;
            virtual void apply(gremlinxx::GraphTraversal* trv, gremlinxx::traversal::TraverserSet& traversers);

            using gremlinxx::LimitSupportingStep::set_limit;
            virtual void set_limit(size_t limit) { this->limit.emplace(limit); }

            using gremlinxx::LimitSupportingStep::get_limit;
            virtual std::optional<size_t> get_limit() { return this->limit; }

            inline virtual void add_predicate(std::string key, gremlinxx::P val) { this->predicates.push_back(std::make_pair(key, val)); }

            inline virtual void add_fuzzy(bitgraph::fuzzy_t fuzzy) { this->fuzzies.push_back(fuzzy); }

            inline virtual std::vector<std::pair<std::string, gremlinxx::P>> get_predicates() { return this->predicates; }

            using gremlinxx::TraversalStep::getInfo;
            virtual std::string getInfo() {
                std::stringstream sx;
                sx << "BitGraphVStep(";

                if(!this->predicates.empty()) {
                    for(auto it = predicates.begin(); it != predicates.end(); ++it) {
                        sx << it->first << " " << maelstrom::comparator_names[it->second.comparison] << " " << string_any(it->second.operand);
                        if(it != predicates.end() - 1) sx << ", ";
                    }

                    if(this->limit) sx << ", ";
                }

                if(this->limit) sx << *this->limit;
                
                sx << ")";
                return sx.str();
            }
    };

}