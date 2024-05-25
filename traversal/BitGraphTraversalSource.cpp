#include "bitgraph/traversal/BitGraphTraversalSource.h"
#include "bitgraph/structure/BitGraph.h"
#include "bitgraph/strategy/BitGraphStrategy.h"
#include "gremlinxx/gremlinxx.h"

namespace bitgraph {

    BitGraphTraversalSource::BitGraphTraversalSource(BitGraph* gr)
    : gremlinxx::GraphTraversalSource(gr) {
        this->withTypeRegistration(
            std::type_index(typeid(std::string)), gr->get_string_dtype()
        );

        this->withStrategy(bitgraph::BitGraphStrategy);
        this->withStrategy(gremlinxx::LimitSupportingStrategy);
    }
}