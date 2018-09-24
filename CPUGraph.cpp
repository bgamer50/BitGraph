#include "CPUGraph.h"
#include "CPUGraphTraversalSource.h"

// Create an empty CPU Graph
CPUGraph::CPUGraph()
: Graph() {
	//vertices = new std::vector<BitVertex>();
}

std::vector<Vertex*> CPUGraph::vertices() {
	return vertex_list;
}

std::vector<Edge*> CPUGraph::edges() {
	return edge_list;
}

GraphTraversalSource* CPUGraph::traversal() {
	CPUGraph* ref = this;
	return new CPUGraphTraversalSource(ref);
}

Vertex* CPUGraph::add_vertex(std::string label) {
	Vertex* v = new BitVertex(NEXT_VERTEX_ID_CPU(), label);
	vertex_list.push_back(v);
	return v;
}

Vertex* CPUGraph::add_vertex() {
	Vertex* v = new BitVertex(NEXT_VERTEX_ID_CPU());
	vertex_list.push_back(v);
	return v;
}