#pragma once

#include "maelstrom/algorithms/similarity.h"
#include <string>
#include <optional>

namespace bitgraph {

    struct fuzzy_t {
        maelstrom::vector embeddings;
        std::string emb_name;
        size_t emb_stride;
        std::optional<double> match_threshold;
        std::optional<size_t> count;
        maelstrom::similarity_t similarity_metric;
    };

}