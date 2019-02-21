#ifndef INDEX_STEP_H
#define INDEX_STEP_H

#define INDEX_STEP 0x12

#include <boost/any.hpp>
#include "TraversalStep.h"

class IndexStep : public TraversalStep {
    private:
        std::string key;
        boost::any value;
    public:
        IndexStep(std::string key, boost::any value)
        : TraversalStep(MAP, INDEX_STEP) {
            this->key = key;
            this->value = value;
        }

        std::string get_key() {return key;}
        boost::any get_value() {return value;}
};

#endif