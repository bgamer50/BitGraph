#ifndef GPU_GRAPH_TRAVERSAL_SOURCE_H
#define GPU_GRAPH_TRAVERSAL_SOURCE_H

#include "traversal/GraphTraversalSource.h"
class GPUGraph;

class GPUGraphTraversalSource : public GraphTraversalSource {
	public:
		GPUGraphTraversalSource(GPUGraph* gr);
};

#include "structure/GPUGraph.h"
#include "strategy/TraversalStrategy.h"
#include "strategy/GPUGraphStrategy.h"

GPUGraphTraversalSource::GPUGraphTraversalSource(GPUGraph* gr)
: GraphTraversalSource(static_cast<Graph*>(gr)) {
			this->strategies.push_back([](std::vector<TraversalStep*>& t) {
				gpugraph_strategy(t);
			});
		}

#endif