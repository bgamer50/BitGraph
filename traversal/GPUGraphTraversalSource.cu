#include "traversal/GPUGraphTraversalSource.cuh"

#include "structure/GPUGraph.cuh"
#include "strategy/TraversalStrategy.h"
#include "strategy/GPUGraphStrategy.cuh"

GPUGraphTraversalSource::GPUGraphTraversalSource(GPUGraph* gr)
: GraphTraversalSource(static_cast<Graph*>(gr)) {
	this->strategies.push_back([](std::vector<TraversalStep*>& t) {
		gpugraph_strategy(t);
	});
}