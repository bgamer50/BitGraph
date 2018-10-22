#include <stdio.h>
#include "CPUGraph.h"
#include "P.h"
#include "__.h"
#include <string>
#include <algorithm>
#include <iostream>

int main(int argc, char* argv[]) {
	CPUGraph graph;
	std::string explanation = graph.traversal()->addV()->has("a", "b")->explain();
	printf("%s\n", explanation.c_str());

	graph.traversal()->addV()->addV()->addV()->V()->iterate();
	/*
	graph.traversal()->addE()
		->from(__->V()->hasId(1))
		->to(__->V()->hasId(1))
		->V().hasId(1)
		->addE()
		->to(__->V()->hasId(1))
		->iterate();
	*/
	graph.traversal()->addE("basic_edge")->from(graph.vertices()[0])->to(graph.vertices()[1])->iterate();

	cout << graph.vertices().size();
	for_each(graph.vertices().begin(), graph.vertices().end(), [](Vertex* v){ cout << *((int*)v->id()) << "\n"; });
	cout << "---------\n";

	std::vector<BitEdge*> edges = ((BitVertex*)graph.vertices()[0])->edges(OUT);
	for_each(edges.begin(), edges.end(), [](BitEdge* edg){ cout << *((int*)edg->id()) << "\n"; });
}