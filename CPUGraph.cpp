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

Edge* CPUGraph::add_edge(BitVertex* from_vertex, BitVertex* to_vertex, std::string label) {
	BitEdge* new_edge = new BitEdge(NEXT_EDGE_ID_CPU(), from_vertex, to_vertex, label);
	from_vertex->addEdge(new_edge, OUT);
	to_vertex->addEdge(new_edge, IN);
	edge_list.push_back(new_edge);
	return new_edge;
}