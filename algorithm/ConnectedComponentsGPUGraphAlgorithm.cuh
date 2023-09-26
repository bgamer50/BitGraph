#pragma once

#include "structure/memory/ThrustUtils.cuh"
#include "algorithm/GPUGraphAlgorithm.h"
#include "gremlinxx/gremlinxx.h"

__global__ void k_cc(size_t N, size_t* row_ptr, size_t* col_ptr, size_t* old_cc, size_t* new_cc);
__global__ void k_sub(size_t N, size_t* A, size_t* B);
__global__ void k_cc_init(size_t N, size_t* cc);

class GPUGraph;

class ConnectedComponentsGPUGraphAlgorithm : public GPUGraphAlgorithm {
    private:
        Direction direction = BOTH;

    public:
        static const inline std::string OPTION_DIRECTION = "DIRECTION";
        static const inline std::string OUTPUT_COMPONENTS = "COMPONENTS";

        virtual std::unordered_map<std::string, boost::any> exec(GPUGraph* graph);

        virtual GPUGraphAlgorithm* option(std::string opt, boost::any value) {
            if(opt == OPTION_DIRECTION) {
                this->direction = boost::any_cast<Direction>(value);
            } else {
                throw std::runtime_error("Invalid option " + opt);
            }

            return this;
        }
};

