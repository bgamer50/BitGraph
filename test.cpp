#include <stdio.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <functional>
#include <boost/any.hpp>
#include "traversal/P.h"
#include "structure/Vertex.h"
#include "traversal/GraphTraversal.h"
#include "structure/CPUGraph.h"

std::vector<std::function<void()>> tests;

int main(int argc, char* argv[]) {
	tests.push_back([]() {
		CPUGraph graph;
		auto explanation = graph.traversal()->addV()->has("a", "b")->explain();
		printf("%s\n", explanation.c_str());

		graph.traversal()->addV()->addV()->addV()->iterate();
		printf("Added 3 vertices to the graph.\n");
		graph.traversal()->V()->property("a", std::string("b"))->iterate();

		std::vector<Vertex*> vertices;
		std::for_each(graph.vertices().begin(), graph.vertices().end(), [&vertices](Vertex* v) {
			vertices.push_back(v);
		});

		VertexProperty<boost::any>* p = vertices.front()->property("a");
		std::cout << boost::any_cast<uint64_t>(vertices.front()->id()) << " Set property " << p->key() << " to " << boost::any_cast<std::string>(p->value()) << "\n";
		
		p = vertices[1]->property("a");
		std::cout << boost::any_cast<uint64_t>(vertices[1]->id()) << " Set property " << p->key() << " to " << boost::any_cast<std::string>(p->value()) << "\n";

		p = vertices[2]->property("a");
		std::cout << boost::any_cast<uint64_t>(vertices[2]->id()) << " Set property " << p->key() << " to " << boost::any_cast<std::string>(p->value()) << "\n";

		std::cout << "size: " << graph.vertices().size() << "\n";

		graph.traversal()->V()->has("a", std::string("b"))->forEachRemaining([](boost::any& v){
			Vertex* w = boost::any_cast<Vertex*>(v);
			std::cout << "The Vertex with id " << boost::any_cast<uint64_t>(w->id()) << " has property a = b\n";			
		});

		/*
		graph.traversal()->addE()
			->from(__->V()->hasId(1))
			->to(__->V()->hasId(1))
			->V().hasId(1)
			->addE()
			->to(__->V()->hasId(1))
			->iterate();
		*/
		graph.traversal()->addE("basic_edge")->from(vertices[0])->to(vertices[1])->iterate();
		cout << "Edge successfully added!\n";

		std::cout << graph.vertices().size() << "\n";
		for_each(vertices.begin(), vertices.end(), [](Vertex* v){ std::cout << boost::any_cast<uint64_t>(v->id()) << " "; });
		std::cout << "\n---------\n";

		std::vector<BitEdge*> edges = ((BitVertex*)graph.vertices().front())->edges(OUT);
		cout << "# edges: " << edges.size() << "\n";
		for_each(edges.begin(), edges.end(), [](BitEdge* edg) { 
			uint64_t edge_id = boost::any_cast<uint64_t>(edg->id());
			uint64_t v1_id = boost::any_cast<uint64_t>(edg->outV()->id());
			uint64_t v2_id = boost::any_cast<uint64_t>(edg->inV()->id());
			std::cout << edge_id << ": " << v1_id << "->" << v2_id << "\n"; 
		});

	});

	for_each(tests.begin(), tests.end(), [](auto n){
		try {
			n();
		} catch(const std::exception & err) { cout << err.what() << "\n"; return -1; }
	});
}
