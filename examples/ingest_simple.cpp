#include <iostream>
#include <cstdio>

#include "structure/CPUGraph.h"
#include "traversal/GraphTraversal.h"
#include "Q.h"
#include "util/C.h"
#include "GPUGraphTraversal.h"

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

        if(0 == names.count(std::string(id1))) v1 = boost::any_cast<Vertex*>(g->addV(LABEL_V)->property(NAME, std::string(id1))->next());
        else v1 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id1))->next());
        names.insert(std::string(id1));
        std::cout << boost::any_cast<uint64_t>(v1->id()) << " " << boost::any_cast<std::string>(v1->property(NAME)->value()) << "\n";
        
        if(0 == names.count(std::string(id2))) v2 = boost::any_cast<Vertex*>(g->addV(LABEL_V)->property(NAME, std::string(id2))->next());
        else v2 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id2))->next());
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

    g->V()->in()->forEachRemaining([](boost::any& v) {
        Vertex* vtx = boost::any_cast<Vertex*>(v);
        std::cout << "id: " << boost::any_cast<std::string>(vtx->property(NAME)->value()) << "\n";
    });

    std::cout << "count: " << graph.edges().size() << "\n";
    for(Edge* e : graph.edges()) std::cout << boost::any_cast<std::string>(e->outV()->property(NAME)->value()) << " -> " << boost::any_cast<std::string>(e->inV()->property(NAME)->value()) << "\n";

    try {
        g->V()->property("cc", both()->id())->forEachRemaining([](boost::any& w) {
            Vertex* v = boost::any_cast<Vertex*>(w);
            boost::any a = v->property("cc")->value();
            std::cout << a.type().name() << "\n";
            std::cout << boost::any_cast<std::string>(v->property(NAME)->value()) << "/" << boost::any_cast<uint64_t>(v->id()) << ": " << boost::any_cast<uint64_t>(v->property("cc")->value()) << "\n";
        });
    } catch(const std::exception& err) {
        std::cout << err.what() << "\n";
        return -1;
    }

    try {
        GPUGraphTraversal* gpu_trv = dynamic_cast<GPUGraphTraversal*>(dynamic_cast<CPUGraphTraversalSource*>(graph.traversal())->withGPU()->V()->id());
        gpu_trv->filter(Q<uint64_t>::eq((uint64_t)1, gpu_trv))->iterate();
    } catch(const std::exception& err) {
        std::cout << err.what() << "\n";
        return -1;
    }

    try {
        std::cout << "min id: " << boost::any_cast<uint64_t>(graph.traversal()->V()->id()->min(C<uint64_t>::compare())->next()) << "\n";
        dynamic_cast<CPUGraphTraversalSource*>(graph.traversal())->withGPU()->V()->id()->min(C<uint64_t>::compare())->iterate();
    } catch(const std::exception& err) {
        std::cout << err.what() << "\n";
        return -1;
    }

    try {
        std::cout << "coalesce:" << std::endl;
        graph.traversal()->V()->coalesce({id(), id()})->forEachRemaining([](boost::any& i) {
            uint64_t id = boost::any_cast<uint64_t>(i);
            std::cout << id << std::endl;
        });
        std::cout << std::endl;
    } catch(const std::exception& err) {
        std::cout << err.what() << std::endl;
        return -1;
    }
}
