#pragma once

#define HAS_WITH_INDEX_STEP 0x13

#include <boost/any.hpp>
#include "gremlinxx/gremlinxx.h"

class HasWithIndexStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    public:
        HasWithIndexStep(std::string key, boost::any value);

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers);
};
