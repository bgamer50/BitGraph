#pragma once

#include "traversal/GraphTraversalSource.h"

class CPUGraph;

class CPUGraphTraversalSource : public GraphTraversalSource {
	public:
		CPUGraphTraversalSource(CPUGraph* gr);
};

#include "structure/CPUGraph.h"
#include "strategy/TraversalStrategy.h"
#include "strategy/BitGraphStrategy.h"

CPUGraphTraversalSource::CPUGraphTraversalSource(CPUGraph* gr)
	: GraphTraversalSource(gr) {
		this->strategies.push_back([this](std::vector<TraversalStep*>& t) {
			bitgraph_strategy(static_cast<CPUGraph*>(this->getGraph()), t);
		});
}
