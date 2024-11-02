#pragma once

#include "maelstrom/containers/vector.h"

namespace bitgraph {
    maelstrom::vector search_embedding_index_knn(maelstrom::vector& emb, std::any& emb_index, maelstrom::vector& query, size_t emb_stride, size_t k);
}