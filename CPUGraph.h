#ifndef CPU_GRAPH_H
#define CPU_GRAPH_H

#include "Graph.h"
#include "BitVertex.h"
#include <string>
#include <vector>

class CPUGraph : public Graph {
	private:
		std::vector<Vertex*> vertex_list;
		std::vector<Edge*> edge_list;
	public:

		/*
			Main constructor for the CPUGraph.
		*/
		CPUGraph();

		/*
			The vector? containing the CPUGraph's vertices.
			May want to change this in the future.
		*/
		std::vector<Vertex*> vertices();

		/*
			The vector? containing the CPUGraph's edges.
		*/
		std::vector<Edge*> edges();

		/*
			Adds a new Vertex (w/label) to this CPUGraph.
			Currently not part of the higher-level api.
		*/
		Vertex* add_vertex(std::string label);

		/*
			Adds a new Vertex (w/o label) to this CPUGraph.
			Currently not part of the higher-level api.
		*/
		Vertex* add_vertex();

		/*
			Get a traversal source for this CPUGraph.
		*/
		GraphTraversalSource* traversal();
}; 

#define NEXT_VERTEX_ID_CPU() ( vertex_list.size() == 0 ? 0 : (*((uint64_t*)vertex_list.back()->id()) + 1) )

#endif