#include <vector>
#include <any>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <functional>

#include <cuda_profiler_api.h>

#include "bitgraph/structure/BitGraph.h"
#include "gremlinxx/gremlinxx.h"
#include "gremlinxx/gremlinxx_utils.h"
#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/arange.h"

#define LABEL_V "basic_vertex"
#define LABEL_E "basic_edge"
#define NAME "name"

void set_vertex_properties(bitgraph::BitGraph& graph, std::string name, maelstrom::vector& keys, maelstrom::vector& values, maelstrom::storage storage=maelstrom::MANAGED, maelstrom::dtype_t dtype=maelstrom::float64) {
    graph.declare_vertex_property(
        name,
        storage,
        dtype,
        graph.num_vertices()
    );

    graph.set_vertex_properties(
        name,
        keys,
        values
    );
}

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
    std::string filename_nodelist = argv[1];
    std::string filename_edgelist = argv[2];
    std::string processor = argv[3];
    size_t tries = std::atol(argv[4]);
    
    size_t expected_num_nodes = (argc >= 6) ? std::atol(argv[5]) : static_cast<size_t>(20E3);
    size_t expected_num_edges = (argc >= 7) ? std::atol(argv[6]) : static_cast<size_t>(1E6);

    std::cout << "Allocating storage for " << expected_num_nodes << " nodes" << std::endl;
    std::cout << "Allocating storage for " << expected_num_edges << " edges" << std::endl;

    // Expected nodelist format is name, node props
    // Expected edgelist format is time, src, dst, edge props

    std::fstream f_nodes;
    f_nodes.open(filename_nodelist, std::ios::in);

    std::vector<uint32_t> source;
    source.reserve(expected_num_edges);
    std::vector<uint32_t> destination;
    destination.reserve(expected_num_edges);

    auto start = std::chrono::system_clock::now();

    std::unordered_map<std::string, uint32_t> names;
    names.reserve(expected_num_nodes);
    
    std::vector<std::any> types;
    types.reserve(expected_num_nodes);

    std::vector<std::any> continents;
    continents.reserve(expected_num_nodes);

    std::vector<std::any> regions;
    regions.reserve(expected_num_nodes);

    std::vector<double> lats;
    lats.reserve(expected_num_nodes);

    std::vector<double> lons;
    lons.reserve(expected_num_nodes);
    
    std::string name;
    std::string line;
    size_t v_id;
    std::getline(f_nodes, line, '\n'); line.clear(); // skip header
    while(!f_nodes.eof()) {
        std::getline(f_nodes, line, '\n');
        
        std::stringstream sx(line, std::ios::in);
        std::string entry;
        std::vector<std::string> entries;
        entries.reserve(6);
        while(std::getline(sx, entry, ',')) entries.push_back(std::string(entry));

        if(entries.size() == 6) {
            name = std::move(entries[0]);
            names[name] = v_id;
            ++v_id;

            auto dtype = graph.get_string_dtype();

            types.push_back(dtype.serialize(entries[1]));
            continents.push_back(dtype.serialize(entries[2]));
            regions.push_back(dtype.serialize(entries[3]));

            lons.push_back(std::atof(entries[4].c_str()));
            lats.push_back(std::atof(entries[5].c_str()));
        }
        
    }

    std::cout << "File read complete." << std::endl;
    f_nodes.close();

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

    set_vertex_properties(graph, NAME, m_vertices, m_values, maelstrom::DEVICE, graph.get_string_dtype());
    m_vertices.clear();
    m_values.clear();

    auto vrange = maelstrom::arange(maelstrom::DEVICE, static_cast<size_t>(names.size())).astype(graph.get_vertex_dtype());

    auto m_types = maelstrom::make_vector_from_anys(maelstrom::PINNED, graph.get_string_dtype(), types);
    set_vertex_properties(graph, "type", vrange, m_types, maelstrom::MANAGED, graph.get_string_dtype());
    m_types.clear();

    auto m_continents = maelstrom::make_vector_from_anys(maelstrom::PINNED, graph.get_string_dtype(), continents);
    set_vertex_properties(graph, "continent", vrange, m_continents, maelstrom::MANAGED, graph.get_string_dtype());
    m_continents.clear();

    auto m_regions = maelstrom::make_vector_from_anys(maelstrom::PINNED, graph.get_string_dtype(), regions);
    set_vertex_properties(graph, "region", vrange, m_regions, maelstrom::MANAGED, graph.get_string_dtype());
    m_regions.clear();

    auto m_lons = maelstrom::vector(maelstrom::PINNED, maelstrom::float64, lons.data(), lons.size(), false);
    set_vertex_properties(graph, "longitude", vrange, m_lons, maelstrom::MANAGED, maelstrom::float64);
    m_lons.clear();

    auto m_lats = maelstrom::vector(maelstrom::PINNED, maelstrom::float64, lats.data(), lats.size(), false);
    set_vertex_properties(graph, "latitude", vrange, m_lats, maelstrom::MANAGED, maelstrom::float64);
    m_lats.clear();

    vrange.clear();
    std::cout << "Set vertex properties" << std::endl;

    std::fstream f_edges;
    f_edges.open(filename_edgelist, std::ios::in);

    // read the edgelist
    std::vector<std::pair<uint32_t, uint32_t>> edge_info;
    edge_info.reserve(expected_num_edges);

    std::chrono::year_month_day start_ymd{
        std::chrono::year(2019),
        std::chrono::month(1),
        std::chrono::day(1)
    };
    std::vector<uint32_t> times; // days since start (1/1/2019)
    times.reserve(expected_num_edges);

    std::vector<std::any> flight_numbers;
    flight_numbers.reserve(expected_num_edges);

    std::vector<std::any> planes;
    planes.reserve(expected_num_edges);

    std::getline(f_edges, line, '\n'); line.clear(); // skip header
    size_t e_id;
    while(!f_edges.eof()) {
        std::getline(f_edges, line, '\n');

        std::string entry;
        std::vector<std::string> entries;
        entries.reserve(4);

        std::stringstream sx(line);
        while(std::getline(sx, entry, ',')) entries.push_back(std::string(entry));

        if(entries.size() == 5) {
            uint32_t src = names[entries[1]];
            uint32_t dst = names[entries[2]];
            edge_info.push_back(std::make_pair(src, dst));
            ++e_id;
            if(e_id % 1000 == 0) std::cout << e_id << std::endl;

            source.push_back(src);
            destination.push_back(dst);

            /*
            std::tm date_info;
            date_info.tm_year = std::atoi(entries[0].substr(0, 4).c_str()) - 1900;
            date_info.tm_mon = std::atoi(entries[0].substr(5, 2).c_str()) - 1;
            date_info.tm_mday = std::atoi(entries[0].substr(8, 2).c_str());
            auto timestamp = std::mktime(&date_info);
            */
            std::chrono::year_month_day ymd{
                std::chrono::year(std::atoi(entries[0].substr(0, 4).c_str())),
                std::chrono::month(std::atoi(entries[0].substr(5, 2).c_str())),
                std::chrono::day(std::atoi(entries[0].substr(8, 2).c_str()))
            };
            auto time_diff = std::chrono::sys_days{ymd} - std::chrono::sys_days{start_ymd};

            times.push_back(
                static_cast<uint32_t>(
                    time_diff.count()
                )
            );
            
            flight_numbers.push_back(
                graph.get_string_dtype().serialize(entries[3])
            );

            planes.push_back(
                graph.get_string_dtype().serialize(entries[4])
            );
        }
    }    

    auto source_view = maelstrom::vector(maelstrom::HOST, maelstrom::uint32, source.data(), source.size(), true);
    auto dest_view = maelstrom::vector(maelstrom::HOST, maelstrom::uint32, destination.data(), destination.size(), true);
    graph.add_edges(
        source_view,
        dest_view,
        LABEL_E
    );
    source.clear();
    destination.clear();
    std::cout << "added edges" << std::endl;

    auto erange = maelstrom::arange(maelstrom::DEVICE, static_cast<size_t>(graph.num_edges())).astype(graph.get_edge_dtype());
    
    maelstrom::vector times_copy(maelstrom::PINNED, maelstrom::uint32, times.data(), times.size(), false);
    times.clear();
    graph.declare_edge_property("time", maelstrom::DEVICE, maelstrom::uint32, graph.num_edges());
    graph.set_edge_properties("time", erange, times_copy);
    times_copy.clear();
    std::cout << "set times" << std::endl;

    auto m_flight_numbers = maelstrom::make_vector_from_anys(maelstrom::PINNED, graph.get_string_dtype(), flight_numbers);
    flight_numbers.clear();
    graph.declare_edge_property("flight_number", maelstrom::DEVICE, graph.get_string_dtype(), graph.num_edges());
    graph.set_edge_properties("flight_number", erange, m_flight_numbers);
    m_flight_numbers.clear();
    std::cout << "set flight numbers" << std::endl;

    auto m_planes = maelstrom::make_vector_from_anys(maelstrom::PINNED, graph.get_string_dtype(), planes);
    planes.clear();
    graph.set_edge_properties("plane", erange, m_planes);
    m_planes.clear();
    std::cout << "set plane" << std::endl;
    
    std::cout << "added edge properties" << std::endl;

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Ingest time: " << elapsed.count() << " seconds." << std::endl;

    g->withAdminOption("debug", "True");
    for(size_t r = 0; r < tries; ++r) {
        try {
            start = std::chrono::system_clock::now();

            using gremlinxx::id;
            using gremlinxx::values;
            
            g->V().limit(3).values(NAME).forEachRemaining([](auto b){
                std::cout << std::any_cast<std::string>(b) << std::endl;
            });

            // do traversals
            auto result = g->E().limit(1).values("flight_number").next();
            std::cout << std::any_cast<std::string>(result) << std::endl;
            auto results = g->E().limit(10).values("flight_number").toVector();
            for(auto& a : results) {
                std::cout << std::any_cast<std::string>(a) << std::endl;
            }

            end = std::chrono::system_clock::now();
            elapsed = end-start;
            std::cerr << "CCxx time: " << elapsed.count() << " seconds." << std::endl;

            // print some validation text

        } catch(const std::exception& err) {
            std::cout << "An error occurred: " << err.what() << std::endl;
            return -1;
        }
    }

    return EXIT_SUCCESS;
}