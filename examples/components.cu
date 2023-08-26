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
#include "maelstrom/algorithms/arange.h"

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

    // FIXME hide all of this behind a reader so the user doesn't have to
    // worry about BitGraph implementation details
    std::string filename = argv[1];
    std::string processor = argv[2];
    size_t tries = std::atol(argv[3]);
    FILE* f = fopen(filename.c_str(), "r");

    char id1[100];
    char id2[100];
    std::unordered_map<std::string, uint32_t> names;
    
    size_t expected_num_edges = (argc >= 5) ? std::atol(argv[4]) : static_cast<size_t>(1E6);
    std::cout << "Allocating storage for " << expected_num_edges << " edges" << std::endl;

    std::vector<uint32_t> source;
    source.reserve(expected_num_edges);
    std::vector<uint32_t> destination;
    destination.reserve(expected_num_edges);

    int k = 0;
    auto start = std::chrono::system_clock::now();
    while(2 == fscanf(f, "%s %s\n", id1, id2)) {
        try {
            ++k;
            if(k % 1000 == 0) std::cout << k << std::endl;
            //std::cout << id1 << ", " << id2 << "\n";

            if(names.end() == names.find(std::string(id1))) {
                names[std::string(id1)] = names.size();
            }
            
            if(names.end() == names.find(std::string(id2))) {
                names[std::string(id2)] = names.size();
            }

            source.push_back(
                names[std::string(id1)]
            );
            destination.push_back(
                names[std::string(id2)]
            );
        } catch(const std::exception& err) {
            std::cout << err.what() << "\n";
            return -1;
        }
    }
    std::cout << "File read complete." << std::endl;
    fclose(f);

    graph.add_vertices(names.size());
    std::cout << "Vertices added." << std::endl;

    std::vector<uint32_t> h_vertices;
    h_vertices.reserve(names.size());
    std::vector<std::any> h_values;
    h_values.reserve(names.size());
    for(auto p : names) {
        h_values.push_back(graph.get_string_dtype().serialize(p.first));
        h_vertices.push_back(p.second);
    }

    maelstrom::vector m_vertices(
        maelstrom::HOST,
        graph.get_vertex_dtype(),
        h_vertices.data(),
        h_vertices.size(),
        false
    );

    auto m_values = maelstrom::make_vector_from_anys(
        maelstrom::HOST,
        graph.get_string_dtype(),
        h_values
    );

    graph.declare_vertex_property(
        NAME,
        maelstrom::DEVICE,
        graph.get_string_dtype(),
        graph.num_vertices()
    );

    graph.set_vertex_properties(
        NAME,
        m_vertices,
        m_values
    );
    std::cout << "Set vertex properties" << std::endl;

    auto source_view = maelstrom::vector(maelstrom::HOST, maelstrom::uint32, source.data(), source.size(), true);
    auto dest_view = maelstrom::vector(maelstrom::HOST, maelstrom::uint32, destination.data(), destination.size(), true);
    graph.add_edges(
        source_view,
        dest_view,
        LABEL_E
    );

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Ingest time: " << elapsed.count() << " seconds." << std::endl;

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

    //g->withAdminOption("debug", "True");
    for(size_t r = 0; r < tries; ++r) {
        try {
            //cudaProfilerStart();
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
            //cudaProfilerStop();
            elapsed = end-start;
            std::cerr << "CCxx time: " << elapsed.count() << " seconds." << std::endl;

            size_t n_components = std::any_cast<size_t>(g->V().values("cc").dedup().count().next());
            std::cout << n_components << " components!" << std::endl;

        } catch(const std::exception& err) {
            std::cout << "An error occurred: " << err.what() << std::endl;
            return -1;
        }
    }

    return EXIT_SUCCESS;
}