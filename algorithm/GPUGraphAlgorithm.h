#ifndef GPU_GRAPH_ALGORITHM_H
#define GPU_GRAPH_ALGORITHM_H

#include <string>
#include <unordered_map>
#include <boost/any.hpp>

#include "structure/GPUGraph.cuh"

// somewhat similar to VertexProgram
class GPUGraphAlgorithm {
    public:
        virtual std::unordered_map<std::string, boost::any> exec(GPUGraph* graph) = 0;
        virtual GPUGraphAlgorithm* option(std::string option, boost::any value) = 0;
};

#endif