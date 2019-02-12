#ifndef CPU_GRAPH_H
#define CPU_GRAPH_H

#include "Graph.h"
#include <string>
#include <vector>
class BitVertex;

class CPUGraph : public Graph {
	private:
		std::vector<Vertex*> vertex_list;
		std::vector<Edge*> edge_list;
		uint64_t next_edge_id = 0;
	public:

		/*
			Main constructor for the CPUGraph.
		*/
		CPUGraph(): Graph() {}

		/*
			The vector? containing the CPUGraph's vertices.
			May want to change this in the future.
		*/
		std::vector<Vertex*> vertices() { return vertex_list; }

		/*
			The vector? containing the CPUGraph's edges.
		*/
		std::vector<Edge*> edges() { return edge_list; }

		/*
			Adds a new Vertex (w/label) to this CPUGraph.
		*/
		virtual Vertex* add_vertex(std::string label);

		/*
			Adds a new Vertex (w/o label) to this CPUGraph.
		*/
		virtual Vertex* add_vertex();

		/*
			Adds a new Edge to this CPUGraph.
		*/
		virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label);

		/*
			Get a traversal source for this CPUGraph.
		*/
		virtual GraphTraversalSource* traversal();
};

#define NEXT_VERTEX_ID_CPU() ( vertex_list.size() == 0 ? 0 : (boost::any_cast<uint64_t>(vertex_list.back()->id()) + 1) )
#define NEXT_EDGE_ID_CPU() ((uint64_t)( next_edge_id++ ))

#include "CPUGraphTraversalSource.h"
#include "BitVertex.h"

/*
	Get a traversal source for this CPUGraph.
*/
GraphTraversalSource* CPUGraph::traversal() {
	CPUGraph* ref = this;
	return new CPUGraphTraversalSource(ref);
}

/*
	Adds a new Vertex (w/label) to this CPUGraph.
	Currently not part of the higher-level api.
*/
Vertex* CPUGraph::add_vertex(std::string label) {
	Vertex* v = new BitVertex(NEXT_VERTEX_ID_CPU(), label);
	vertex_list.push_back(v);
	return v;
}

/*
	Adds a new Vertex (w/o label) to this CPUGraph.
*/
Vertex* CPUGraph::add_vertex() {
	Vertex* v = new BitVertex(NEXT_VERTEX_ID_CPU());
	vertex_list.push_back(v);
	return v;
}

/*
	Adds a new Edge to this CPUGraph.
*/
Edge* CPUGraph::add_edge(Vertex* out, Vertex* in, std::string label) {
	BitVertex* from_vertex = dynamic_cast<BitVertex*>(out);
	BitVertex* to_vertex = dynamic_cast<BitVertex*>(in);

	BitEdge* new_edge = new BitEdge(NEXT_EDGE_ID_CPU(), from_vertex, to_vertex, label);
	from_vertex->addEdge(new_edge, OUT);
	to_vertex->addEdge(new_edge, IN);
	edge_list.push_back(new_edge);
	return new_edge;
}

#endif