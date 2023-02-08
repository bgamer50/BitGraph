#pragma once

#include <assert.h>
#include <vector>

namespace bitgraph {
    namespace test {
        
        template <typename T>
        void assert_vector_equals(std::vector<T>& left, std::vector<T>& right) {
            assert( left.size() == right.size() );
            for(size_t k = 0; k < left.size(); ++k) {
                assert( left[k] == right[k] );
            }
        }

    }
}