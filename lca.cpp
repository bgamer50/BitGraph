/*
Based on the Gremlin recipe for finding the lowest common ancestor in a tree.
Written by Alexandria Barghi
*/

#include "structure/CPUGraph.h"
#define NAME "name"

int main(int charc, char* argv[]) {
    // Create a blank CPU Graph
    CPUGraph graph;
    auto g = graph.traversal()->withAdminOption("verbose", "True");

    g->addV()->property(NAME, "A")->as("a")->
           addV()->property(NAME, "B")->as("b")->
           addV()->property(NAME, "C")->as("c")->
           addV()->property(NAME, "D")->as("d")->
           addV()->property(NAME, "E")->as("e")->
           addV()->property(NAME, "F")->as("f")->
           addV()->property(NAME, "G")->as("g")->
           addE("hasParent")->from("a")->to("b")->
           addE("hasParent")->from("b")->to("c")->
           addE("hasParent")->from("d")->to("c")->
           addE("hasParent")->from("c")->to("e")->
           addE("hasParent")->from("e")->to("f")->
           addE("hasParent")->from("g")->to("f")->iterate();

    std::cout << "graph created" << std::endl;

    g->V()->values(NAME)->forEachRemaining([](boost::any& v){
        auto c = boost::any_cast<const char*>(v);
        std::cout << c << std::endl;
        });
}