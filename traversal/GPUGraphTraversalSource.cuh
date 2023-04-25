#pragma once

#include "traversal/GraphTraversalSource.h"
class GPUGraph;

class GPUGraphTraversalSource : public GraphTraversalSource {
	public:
		GPUGraphTraversalSource(GPUGraph* gr);
};
