#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <unordered_set>

#include "traversal/GraphTraversal.h"
#include "structure/Graph.h"
#include "traversal/CPUGraphTraversal.h"
#include "structure/CPUGraph.h"
#include "util/C.h"

#define LABEL_V "basic_vertex"
#define LABEL_E "basic_edge"
#define NAME "name"

int main(int argc, char* argv[]) {
    CPUGraph graph;

    GraphTraversalSource* g = graph.traversal();
    Vertex* v1 = boost::any_cast<Vertex*>(g->addV()->next());
    Vertex* v2 = boost::any_cast<Vertex*>(g->addV()->next());
    Vertex* v3 = boost::any_cast<Vertex*>(g->addV()->next());
    
    g->addE(LABEL_E)->from(v1)->to(v2)->iterate();
    g->addE(LABEL_E)->from(v2)->to(v3)->iterate();
    std::cout << g->V()->repeat(__->out())->id()->toVector().size() << std::endl;
    /*
    g->V()->repeat(__->out())->times(1)->id()->forEachRemaining([](boost::any v){
        std::cout << boost::any_cast<uint64_t>(v) << std::endl;
    });
    */

    /*
    Expected output:
        0
        2
    */
}