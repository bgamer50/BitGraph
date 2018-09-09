#ifndef CPU_GRAPH_H
#define CPU_GRAPH_H

#include "Graph.h"
#include "BitVertex.h"

class CPUGraph : public Graph {
	private:
		std::vector<Vertex> vertex_list;
		std::vector<Edge> edge_list;
	public:
		CPUGraph();
		std::vector<Vertex> vertices();
		std::vector<Edge> edges();
		GraphTraversalSource* traversal();
}; 

#endif