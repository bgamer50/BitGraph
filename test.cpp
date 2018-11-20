#include <stdio.h>
#include <string.h>
#include "P.h"
#include <string>
#include <algorithm>
#include <iostream>
#include <functional>
#include "Vertex.h"
#include "GraphTraversal.h"
#include "CPUGraph.h"

std::vector<std::function<void()>> tests;

int main(int argc, char* argv[]) {
	tests.push_back([]() {
		CPUGraph graph;
		auto explanation = graph.traversal()->addV()->has("a", "b")->explain();
		printf("%s\n", explanation.c_str());

		Vertex* v_has_a_b = graph.traversal()->addV()->has("a", "b")->next();

		graph.traversal()->addV()->addV()->addV()->iterate();
		printf("Added 3 vertices to the graph.\n");
		graph.traversal()->V()->property("a", "b")->iterate();

		auto p = graph.vertices()[0]->property("a");
		printf("Set property %s to %s on all vertices.\n", p->key().c_str(), static_cast<std::string*>(p->value())->c_str());
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
		printf("Edge successfully added!\n");

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
