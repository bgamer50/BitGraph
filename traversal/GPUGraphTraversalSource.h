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

CPUGraphTraversalSource::CPUGraphTraversalSource(CPUGraph* gr)
	: GraphTraversalSource(gr) {
		this->strategies.push_back([this](std::vector<TraversalStep*>& t) {
			gpugraph_strategy(static_cast<GPUGraph*>(this->getGraph()), t);
		});
}

#endif