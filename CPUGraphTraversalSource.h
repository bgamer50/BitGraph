#ifndef CPU_GRAPH_TRAVERSAL_SOURCE_H
#define CPU_GRAPH_TRAVERSAL_SOURCE_H

#include "GraphTraversalSource.h"
#include "CPUGraph.h"

class CPUGraphTraversalSource : public GraphTraversalSource {
	public:
		CPUGraphTraversalSource(CPUGraph* gr);
		GraphTraversal<Vertex, Vertex>* V();
		GraphTraversal<Vertex, Vertex>* V(Vertex* v);
		GraphTraversal<Edge, Edge>* E();
		GraphTraversal<Vertex, Vertex>* addV();
		GraphTraversal<Vertex, Vertex>* addV(std::string label);
		GraphTraversal<Edge, Edge>* addE(std::string label);
};

#endif