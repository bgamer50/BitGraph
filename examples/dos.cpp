/*
Written by Alexandria Barghi
Degrees of Separation
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

    std::string start_vertex_name = "100";
    std::string filename = argv[1];
    std::string processor = argv[2];
    size_t tries = std::atol(argv[3]);
    FILE* f = fopen(filename.c_str(), "r");

    char id1[10];
    char id2[10];
    std::unordered_set<std::string> names;
    int k = 0;
    auto start = std::chrono::system_clock::now();
    while(2 == fscanf(f, "%s %s\n", id1, id2)) {
        ++k;
        if(k % 1000 == 0) std::cout << k << std::endl;
        //std::cout << id1 << ", " << id2 << "\n";
        Vertex* v1;
        Vertex* v2;

        if(0 == names.count(std::string(id1))) v1 = boost::any_cast<Vertex*>(g->addV(LABEL_V)->property(NAME, std::string(id1))->next());
        else v1 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id1))->next());
        names.insert(std::string(id1));
        
        if(0 == names.count(std::string(id2))) v2 = boost::any_cast<Vertex*>(g->addV(LABEL_V)->property(NAME, std::string(id2))->next());
        else v2 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id2))->next());
        names.insert(std::string(id2));
        
        try {
            g->V(v1)->addE(LABEL_E)->to(v2)->iterate();
        } catch(const std::exception& err) {
            std::cout << err.what() << "\n";
            return -1;
        }
    }

    std::cout << "graph created" << std::endl;
    cudaSetDevice(0);
    GPUGraph gpu_graph(graph);
    g = processor == "gpu" ? gpu_graph.traversal() : graph.traversal();

    Vertex* start_vertex = boost::any_cast<Vertex*>(g->V()->has(NAME, start_vertex_name)->next());
    for(size_t r = 0; r < tries; ++r) {
        std::cout << "Calculating 3 degrees of separation from vertex " << start_vertex_name << std::endl;
        start = std::chrono::system_clock::now();
        auto count = boost::any_cast<size_t>(g->V(start_vertex)->both()->dedup()->both()->dedup()->both()->dedup()->count()->next());
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << count << std::endl;
        std::cerr << "dos time: " << elapsed.count() << " seconds." << std::endl;

        std::cout << "Calculating 3 degrees of separation from vertex " << start_vertex_name << " using repeat step" << std::endl;
        start = std::chrono::system_clock::now();
        count = boost::any_cast<size_t>(g->V(start_vertex)->repeat(__->both())->times(3)->dedup()->count()->next());
        end = std::chrono::system_clock::now();
        elapsed = end - start;
        std::cout << count << std::endl;
        std::cerr << "dos time with repeat: " << elapsed.count() << " seconds." << std::endl;
    }
}