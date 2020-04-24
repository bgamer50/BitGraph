#include "traversal/GraphTraversal.h"
#include "structure/CPUGraph.h"
#include "util/gremlin_utils.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/any.hpp>

int main(int argc, char* argv[]) {
    CPUGraph graph;
    GraphTraversalSource* g = graph.traversal();
    g->addV()->property("name","joe")->property("age",30)->property("height", 70)->iterate();
    g->addV()->property("name","bill")->property("age",40)->property("height", 72)->iterate();
    g->addV()->property("name","george")->property("age",25)->property("height", 67)->iterate();

    g->V()->valueMap({"name","age","height"})->by(__->unfold())->where("age", P::gte(30))->forEachRemaining([](boost::any t){
        valuemap_t vm = boost::any_cast<valuemap_t>(t);
        print_valuemap(vm);
    });
}