#include <stdio.h>
#include "CPUGraph.h"
#include "P.h"
#include "__.h"
#include <string>
#include <algorithm>
#include <iostream>
#include <functional>

std::vector<std::function<void()>> tests;

int main(int argc, char* argv[]) {
	tests.push_back([]() {
		CPUGraph graph;
		std::string explanation = graph.traversal()->addV()->has("a", "b")->explain();
		printf("%s\n", explanation.c_str());

		graph.traversal()->addV()->addV()->addV()->iterate();
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


		std::vector<Vertex*> vertices = graph.vertices();
		cout << vertices.size() << "\n";
		for_each(vertices.begin(), vertices.end(), [](Vertex* v){ cout << *((uint64_t*)v->id()) << " "; });
		cout << "\n---------\n";

		std::vector<BitEdge*> edges = ((BitVertex*)graph.vertices()[0])->edges(OUT);
		cout << "# edges: " << edges.size() << "\n";
		for_each(edges.begin(), edges.end(), [](BitEdge* edg){ cout << *((uint64_t*)edg->id()) << ": " << *((uint64_t*)edg->outV()->id()) << "->" << *((uint64_t*)edg->inV()->id()) << "\n"; });

	});

	for_each(tests.begin(), tests.end(), [](auto n){
		try {
			n();
		} catch(const std::exception & err) { cout << err.what() << "\n"; return -1; }			
	});
}