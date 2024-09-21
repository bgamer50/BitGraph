#pragma once

#include "maelstrom/algorithms/similarity.h"
#include <string>

namespace bitgraph {

    struct fuzzy_t {
        maelstrom::vector embeddings;
        std::string emb_name;
        size_t emb_stride;
        double match_threshold;
        maelstrom::similarity_t similarity_metric;
    };

}