#ifndef CPU_GRAPH_TRAVERSAL_SOURCE_H
#define CPU_GRAPH_TRAVERSAL_SOURCE_H

#include "GraphTraversalSource.h"
#include "CPUGraph.h"

class CPUGraphTraversalSource : public GraphTraversalSource {
	public:
		CPUGraphTraversalSource(CPUGraph* gr);
		GraphTraversal* V();
		GraphTraversal* E();
		GraphTraversal* addV();
		GraphTraversal* addV(std::string label);
		GraphTraversal* addE(std::string label);
};

#endif