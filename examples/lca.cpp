/*
Based on the Gremlin recipe for finding the lowest common ancestor in a tree.
Written by Alexandria Barghi
*/

#include "structure/CPUGraph.h"
#include "structure/GPUGraph.h"

#include <string>
#include <fstream>
#include <sstream>

#define NAME "NAME"
#define LABEL_V "basic_vertex"
#define LABEL_E "basic_edge"

std::string get_string(boost::any& b) {
    return std::string(boost::any_cast<const char*>(b));
}

int main(int charc, char* argv[]) {
    // Create a blank CPU Graph
    CPUGraph graph;
    graph.create_index(VERTEX_INDEX, NAME, [](boost::any& a) { 
        std::hash<std::string> hf;
        return hf(boost::any_cast<std::string>(a));
    }, [](boost::any& a, boost::any& b) {
        std::string c = boost::any_cast<std::string>(a);
        std::string d = boost::any_cast<std::string>(b);
        return c == d;
    });
    auto g = graph.traversal();

    std::string nodes_file = argv[1];
    std::string edges_file = argv[2];
    std::string processor = argv[3];
    std::string voi1 = argv[4];
    std::string voi2 = argv[5];

    /*
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
    */

    try {
        std::unordered_set<std::string> names;

        std::ifstream nodes_stream(nodes_file);
        if(!nodes_stream.is_open()) throw std::runtime_error("Could not open file " + nodes_file);

        std::vector<std::string> header;
        if(nodes_stream.good()) {
            std::string header_line;
            std::getline(nodes_stream, header_line);
            auto header_stream = std::stringstream(header_line);

            std::string column;
            while(std::getline(header_stream, column, ',')) header.push_back(column);
        }

        std::string line;
        size_t line_num = 0;
        while(std::getline(nodes_stream, line)) {
            if(line_num % 1000 == 0) std::cout << line_num << std::endl;

            std::string colvalue;
            auto line_stream = std::stringstream(line);
            size_t col_num = 0;
            auto trv = g->addV(LABEL_V);
            while(std::getline(line_stream, colvalue, ',')) {
                trv->property(header[col_num], colvalue);
                ++col_num;
            }
            names.insert(boost::any_cast<std::string>(trv->values(NAME)->next()));
            ++line_num;
        }
        nodes_stream.close();

        // ingest edges
        FILE* f = fopen(edges_file.c_str(), "r");

        char id1[10];
        char id2[10];
        int k = 0;
        auto start = std::chrono::system_clock::now();
        while(2 == fscanf(f, "%s %s\n", id1, id2)) {
            ++k;
            if(k % 1000 == 0) std::cout << k << std::endl;
            //std::cout << id1 << ", " << id2 << "\n";
            Vertex* v1;
            Vertex* v2;

            v1 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id1))->next());
            v2 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id2))->next());
            
            try {
                g->V(v1)->addE(LABEL_E)->to(v2)->iterate();
            } catch(const std::exception& err) {
                std::cout << err.what() << "\n";
                return -1;
            }
        }

        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed = end-start;
        std::cerr << "Ingest time: " << elapsed.count() << " seconds." << std::endl;
    } catch(std::exception& err) {
        std::cout << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "graph created" << std::endl;
    cudaSetDevice(0);
    GPUGraph gpu_graph(graph);
    g = processor == "gpu" ? gpu_graph.traversal() : graph.traversal();

    try {
        start = std::chrono::system_clock::now();
        boost::any lca = 
            g->V()->has(NAME, voi1)->
                repeat(__->out())->emit()->as("x")->
                repeat(__->in())->emit(__->has(NAME, voi2))->
                select("x")->limit(1)->values(NAME)->next();
        end = std::chrono::system_clock::now();
        std::cout << "found the lca!" << std::endl;
        std::cout << "The lowest common ancestor is " + boost::any_cast<std::string>(lca) << std::endl;
        
        elapsed = end - start;
        std::cerr << "lca time: " << elapsed.count() " seconds" << std::endl;
        
    } catch(std::exception& err) {
        std::cout << err.what() << std::endl;
    }
}