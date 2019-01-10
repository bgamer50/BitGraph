#include <iostream>
#include <cstdio>

#include "CPUGraph.h"
#include "GraphTraversal.h"

#define LABEL_V "basic_vertex"
#define LABEL_E "basic_edge"
#define NAME "name"

/*
    Opens the edgelist file provided by the user and create a graph out of it.
*/
int main(int argc, char* argv[]) {
    CPUGraph graph;
    GraphTraversalSource* g = graph.traversal();

    std::string filename = std::string(argv[1]);
    FILE* f = fopen(filename.c_str(), "r");

    char id1[10];
    char id2[10];
    std::set<std::string> names;
    while(2 == fscanf(f, "%s %s\n", id1, id2)) {
        std::cout << id1 << ", " << id2 << "\n";
        Vertex* v1;
        Vertex* v2;

        if(0 == names.count(std::string(id1))) v1 = g->addV(LABEL_V)->property(NAME, std::string(id1))->next();
        else v1 = g->V()->has(NAME, std::string(id1))->next();
        names.insert(std::string(id1));
        std::cout << boost::any_cast<uint64_t>(v1->id()) << " " << boost::any_cast<std::string>(v1->property(NAME)->value()) << "\n";
        
        if(0 == names.count(std::string(id2))) v2 = g->addV(LABEL_V)->property(NAME, std::string(id2))->next();
        else v2 = g->V()->has(NAME, std::string(id2))->next();
        names.insert(std::string(id2));
        std::cout << boost::any_cast<uint64_t>(v2->id()) << " " << boost::any_cast<std::string>(v2->property(NAME)->value()) << "\n";

        std::cout << boost::any_cast<uint64_t>(v1->id()) << " - - " << boost::any_cast<uint64_t>(v2->id()) << "\n";
        
        try {
            g->V(v1)->addE(LABEL_E)->to(v2)->iterate();
        } catch(const std::exception& err) {
            std::cout << err.what() << "\n";
            return -1;
        }
    }

    fclose(f);

    g->V()->in()->forEachRemaining([](void* v) {
        Vertex* vtx = static_cast<Vertex*>(v);
        std::cout << "id: " << boost::any_cast<std::string>(vtx->property(NAME)->value()) << "\n";
    });

    std::cout << "count: " << graph.edges().size() << "\n";
    for(Edge* e : graph.edges()) std::cout << boost::any_cast<std::string>(e->outV()->property(NAME)->value()) << " -> " << boost::any_cast<std::string>(e->inV()->property(NAME)->value()) << "\n";
}
