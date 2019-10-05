#ifndef CPU_GRAPH_TRAVERSAL_H
#define CPU_GRAPH_TRAVERSAL_H

#include <algorithm>
#include <iostream>
#include <boost/any.hpp>

#include "GraphTraversal.h"
#include "GraphTraversalSource.h"

class CPUGraphTraversal : public GraphTraversal {
	public:
		CPUGraphTraversal(GraphTraversalSource* src);

};

#include "CPUGraph.h"

CPUGraphTraversal::CPUGraphTraversal(GraphTraversalSource* src)
	: GraphTraversal(src) {}

#endif
