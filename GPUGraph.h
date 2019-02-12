#include "CPUGraph.h"

/*
    INTEGER converts to uint64_t
    FLOAT converts to double
    STRING converts to std::string
*/
enum PropertyType{INTEGER, FLOAT, STRING};

class GPUGraph: public CPUGraph {
    private:
        std::map<std::string, PropertyType> property_types;

    public:
        GPUGraph(std::map<string, PropertyType> property_type_map) {
            for(auto it = map.begin(); it != map.end(); ++it) {
                this->property_types.emplace(it->first(), it->second());
            }
        }

        /*
            Get a traversal source for this GPUGraph.
        */
        GraphTraversalSource* GPUGraph::traversal() {
            GPUGraph* ref = this;
            return new CPUGraphTraversalSource(ref).withGPU();
        }
};