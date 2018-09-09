#include "CPUGraph.h"
#include "CPUGraphTraversalSource.h"

// Create an empty CPU Graph
CPUGraph::CPUGraph()
: Graph() {
	//vertices = new std::vector<BitVertex>();
}

std::vector<Vertex> CPUGraph::vertices() {
	return vertex_list;
}

std::vector<Edge> CPUGraph::edges() {
	return edge_list;
}

GraphTraversalSource* CPUGraph::traversal() {
	CPUGraph* ref = this;
	return new CPUGraphTraversalSource(ref);
}