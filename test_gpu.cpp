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
    sparse_adj_matrix_t M;
    M.nnz = 7;
    M.col_ptr = {2,0,1,2,0,2,4};
    M.row_ptr = {0,1,4,4,7};
    M.values = {5,1,2,3,1,3,9};

    std::cout << sparse_get(M, 0, 0) << sparse_get(M, 3, 2) << sparse_get(M, 2, 2) << std::endl;
    sparse_set(M, 0, 0, 3.);
    std::cout << sparse_get(M, 0, 0) << sparse_get(M, 3, 2) << sparse_get(M, 2, 2) << std::endl;
    sparse_set(M, 2, 1, 5.);
    std::cout << sparse_get(M, 2, 0) << sparse_get(M, 2, 1) << sparse_get(M, 2, 2) << std::endl;
    sparse_set(M, 0, 0, 0);
    std::cout << sparse_get(M, 0, 0) << std::endl;
    /*
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
    */
}