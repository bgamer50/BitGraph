#pragma once

#include <vector>
#include <memory>

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {

    void bitgraph_strategy(std::vector<std::shared_ptr<gremlinxx::TraversalStep>>& steps);

}