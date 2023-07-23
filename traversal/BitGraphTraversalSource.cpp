#include "bitgraph/traversal/BitGraphTraversalSource.h"
#include "bitgraph/structure/BitGraph.h"

namespace bitgraph {

    BitGraphTraversalSource::BitGraphTraversalSource(BitGraph* gr)
    : gremlinxx::GraphTraversalSource(gr) {
        // TODO strategies, type registrations, etc.		
    }

}