#include <vector>
#include <any>
#include <string>
#include <chrono>
#include <ctime>
#include <unordered_set>

#include <cuda_profiler_api.h>

#include "bitgraph/structure/BitGraph.h"
#include "gremlinxx/gremlinxx.h"
#include "gremlinxx/gremlinxx_utils.h"
#include "maelstrom/containers/vector.h"

#define LABEL_V "basic_vertex"
#define LABEL_E "basic_edge"
#define NAME "name"

int main(int argc, char* argv[]) {
    cudaSetDevice(0);

    bitgraph::BitGraph graph(
        maelstrom::uint32, // vertex dtype
        maelstrom::uint32, // edge dtype
        maelstrom::DEVICE, // structure storage
        maelstrom::MANAGED, // default property storage
        maelstrom::DEVICE // traverser_storage
    );
    auto g = graph.traversal();
    //g->withAdminOption("debug", "True");

    std::string filename = argv[1];
    std::string processor = argv[2];
    size_t tries = std::atol(argv[3]);
    FILE* f = fopen(filename.c_str(), "r");

    char id1[10];
    char id2[10];
    std::unordered_map<std::string, gremlinxx::Vertex> names;
    
    std::vector<uint32_t> source;
    source.reserve(100000);
    std::vector<uint32_t> destination;
    destination.reserve(100000);

    graph.declare_vertex_property(
        NAME,
        maelstrom::DEVICE,
        graph.get_string_dtype(),
        100000
    );

    int k = 0;
    auto start = std::chrono::system_clock::now();
    double add_edge_time = 0.0;
    while(2 == fscanf(f, "%s %s\n", id1, id2)) {
        try {
            ++k;
            if(k % 1000 == 0) std::cout << k << std::endl;
            //std::cout << id1 << ", " << id2 << "\n";

            if(0 == names.count(std::string(id1))) {
                gremlinxx::Vertex v1 = std::any_cast<gremlinxx::Vertex>(
                    g->addV(LABEL_V).property(NAME, std::string(id1)).next()
                );
                names[std::string(id1)] = v1;
            }
            
            if(0 == names.count(std::string(id2))) {
                gremlinxx::Vertex v2 = std::any_cast<gremlinxx::Vertex>(
                    g->addV(LABEL_V).property(NAME, std::string(id2)).next()
                );
                names[std::string(id2)] = v2;
            }

            source.push_back(static_cast<uint32_t>(names[std::string(id1)].id));
            destination.push_back(static_cast<uint32_t>(names[std::string(id2)].id));
        } catch(const std::exception& err) {
            std::cout << err.what() << "\n";
            return -1;
        }
    }

    auto source_view = maelstrom::vector(maelstrom::HOST, maelstrom::uint32, source.data(), source.size(), true);
    auto dest_view = maelstrom::vector(maelstrom::HOST, maelstrom::uint32, destination.data(), destination.size(), true);
    graph.add_edges(
        source_view,
        dest_view,
        LABEL_E
    );

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;
    std::cerr << "Ingest time: " << elapsed.count() << " seconds." << std::endl;
    std::cout << add_edge_time << std::endl;
    //return;

    graph.declare_vertex_property(
        "cc",
        maelstrom::DEVICE,
        maelstrom::uint64
    );

    graph.declare_vertex_property(
        "old_cc",
        maelstrom::DEVICE,
        maelstrom::uint64
    );

    for(size_t r = 0; r < tries; ++r) {
        try {
            cudaProfilerStart();
            start = std::chrono::system_clock::now();

            using gremlinxx::id;
            using gremlinxx::values;
            g->V().property("cc", id()).iterate();
            g->V().property("old_cc", values("cc")).iterate();
            std::cout << "set cc and old cc" << std::endl;

            //gpu_graph.get_property("cc", 13, true);
            //gpu_graph.get_property("old_cc", 13, true);
            //std::cout << "successfully got #13" << std::endl;
            
            size_t diff = 1;
            while(diff > 0) {
                using gremlinxx::_union;
                using gremlinxx::both;
                diff = std::any_cast<size_t>(
                    g->V()
                    .property("old_cc", values("cc"))
                    .property("cc", 
                        _union({both().values("old_cc"), values("old_cc")}).min()
                    )
                    .elementMap({"cc", "old_cc"})
                    .where("cc", gremlinxx::P::neq("old_cc"))
                    .count()
                    .next()
                );
                std::cout << "diff: " << diff << std::endl;
            }
            end = std::chrono::system_clock::now();
            cudaProfilerStop();
            elapsed = end-start;
            std::cerr << "CCxx time: " << elapsed.count() << " seconds." << std::endl;
            std::unordered_set<int> comp_set;
            g->V().values("cc").forEachRemaining([g,&comp_set](std::any& v) {
                int id = std::any_cast<uint64_t>(v);
                comp_set.insert(id);
                //std::cout << id << std::endl;
            });
            std::cout << comp_set.size() << " components!" << std::endl;

        } catch(const std::exception& err) {
            std::cout << err.what() << std::endl;
            return -1;
        }
    }

    fclose(f);
}