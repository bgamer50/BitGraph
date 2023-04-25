#pragma once

#define INDEX_STEP 0x12

#include <boost/any.hpp>
#include "gremlinxx/gremlinxx.h"

class IndexStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    public:
        IndexStep(std::string k, boost::any v);

        std::string get_key() {return key;}
        boost::any get_value() {return value;}

        virtual void apply(GraphTraversal* trv, TraverserSet& traversers);
};

