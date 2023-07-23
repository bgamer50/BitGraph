#pragma once

#include "gremlinxx/gremlinxx.h"

namespace bitgraph {

	class BitGraph;

	class BitGraphTraversalSource : public gremlinxx::GraphTraversalSource {
		public:
			BitGraphTraversalSource(BitGraph* gr);
};

}
