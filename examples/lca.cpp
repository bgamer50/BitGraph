/*
Based on the Gremlin recipe for finding the lowest common ancestor in a tree.
Written by Alexandria Barghi
*/

#include "structure/CPUGraph.h"
#include "structure/GPUGraph.h"
#define NAME "name"

std::string get_string(boost::any& b) {
    return std::string(boost::any_cast<const char*>(b));
}

int main(int charc, char* argv[]) {
    // Create a blank CPU Graph
    CPUGraph graph;
    auto g = graph.traversal();

    try {
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
    } catch(std::exception& err) {
        std::cout << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "graph created" << std::endl;
    GPUGraph gpu_graph(graph);
    g = gpu_graph.traversal();

    try {
        boost::any lca = 
            g->V()->has(NAME, "A")->
                repeat(__->out())->emit()->as("x")->
                repeat(__->in())->emit(__->has(NAME, "D"))->
                select("x")->limit(1)->values(NAME)->next();
        std::cout << "found the lca!" << std::endl;
        std::cout << "The lowest common ancestor of A and D is " + get_string(lca) << std::endl;
        std::cout << ((get_string(lca) == "C") ? "Success!" : "Failure!") << std::endl;
        
    } catch(std::exception& err) {
        std::cout << err.what() << std::endl;
    }
}