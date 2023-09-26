#pragma once

#include "gremlinxx/gremlinxx.h"
#include <string>

namespace bitgraph {

	class BitGraph;

	class BitGraphTraversalSource : public gremlinxx::GraphTraversalSource {
		public:
			BitGraphTraversalSource(BitGraph* gr);
};

}
