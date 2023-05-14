#pragma once

#include <inttypes.h>
#include <vector>
#include <unordered_map>

#include "gremlinxx/gremlinxx.h"
#include "structure/memory/TypeErasure.cuh"

namespace bitgraph {
    namespace traversal {
        
        class GPUTraverserSet {
            private:
                bitgraph::memory::TypeErasedVector traverser_data;
                std::unordered_map<std::string, bitgraph::memory::TypeErasedVector> side_effects;
                std::vector<bitgraph::memory::TypeErasedVector> paths;

                bool persist_paths = false;
            
            public:
                GPUTraverserSet(){}

                /*
                    Returns a TraverserSet containing the traversers on the CPU.
                */
                TraverserSet to_cpu_traversers(GraphTraversal* parent_traversal, StringIndex* string_index=nullptr);


                /*
                    Replaces this set's traversers data with the given CPU traversers
                */
                void from_cpu_traversers(TraverserSet& cpu_traversers, StringIndex* string_index=nullptr);
                
                /*
                    Erases all traversers and their data
                */
                void clear();

                /*
                    Returns a reference to this GPU traverser set's traverser data.
                    Modifying this reference will directly modify the data in this set.
                */
                bitgraph::memory::TypeErasedVector& access_traverser_data();
                
                /*
                    Returns a reference to this GPU traverser set's traverser side
                    effects.  Modifying this reference will directly modify the
                    side effects in this set.
                */
                std::unordered_map<std::string, bitgraph::memory::TypeErasedVector>& access_traverser_side_effects();

                /*
                    Returns a reference to this GPU traverser set's path data.  Modifying
                    this reference will directly modify the path data in this set.
                */
                std::vector<bitgraph::memory::TypeErasedVector>& access_traverser_paths;

                /*
                    Return the number of traversers contained in this gpu traverser set.
                */
                inline size_t size() { return this->traverser_data.size(); }
        };

    }
}