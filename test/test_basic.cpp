#include "gremlinxx/gremlinxx.h"
#include "bitgraph/structure/BitGraph.h"
#include "maelstrom/containers/vector.h"

#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <assert.h>

#include <stdio.h>
#include <any>

using namespace bitgraph;

int main(int argc, char* argv[]) {
    BitGraph bitgraph(
        maelstrom::uint32, // vertex dtype
        maelstrom::uint32, // edge dtype
        maelstrom::DEVICE, // structure storage
        maelstrom::MANAGED, // default property storage
        maelstrom::DEVICE // traverser_storage
    );

    auto g = bitgraph.traversal();
    g->withAdminOption("debug", "True");

    std::fstream fx("data/ds1.txt", std::ios::in);

    std::unordered_map<uint32_t, uint32_t> seen_vertices;

    std::string line;
    g->addE("basic_edge").from(gremlinxx::Vertex{"",0});
    while(getline(fx, line)) {
        // sscanf is just cleaner syntax for this case
        uint32_t src, dst;
        sscanf(line.c_str(), "%u %u", &src, &dst);

        using gremlinxx::Vertex;
        Vertex src_v, dst_v;
        if(seen_vertices.find(src) == seen_vertices.end()) {
            std::cout << "adding " << src << std::endl;
            src_v = std::any_cast<Vertex>(g->addV("basic_vertex").next());
            seen_vertices[src] = src_v.id;
            std::cout << "added" << std::endl;
        } else {
            src_v = std::any_cast<Vertex>(g->V(seen_vertices[src]).next());
        }
        if(seen_vertices.find(dst) == seen_vertices.end()) {
            std::cout << "adding " << dst << std::endl;
            dst_v = std::any_cast<Vertex>(g->addV("basic_vertex").next());
            seen_vertices[dst] = dst_v.id;
            std::cout << "added" << std::endl;
        } else {
            dst_v = std::any_cast<Vertex>(g->V(seen_vertices[dst]).next());
        }

        g->addE("basic_edge").from(src_v).to(dst_v).iterate();
        std::cout << "added edge" << std::endl;
    }

    std::cout << "Get vertex count" << std::endl;
    std::any num_vertices = g->V().count().next();
    std::cout << "Got " << std::any_cast<size_t>(num_vertices) << " vertices" << std::endl;
    assert( std::any_cast<size_t>(num_vertices) == 8 );

    std::cout << "Add property to vertex 16" << std::endl;
    g->V(6).property("bleh", 62).iterate();
    std::cout << "Successfully set property on vertex 6" << std::endl;

    std::cout << "Add property to vertex 7, check output" << std::endl;
    auto v7 = std::any_cast<gremlinxx::Vertex>(
        g->V(7).property("bleh", 66).next()
    );
    std::cout << "Got vertex with id " << v7.id << " and label " << v7.label << std::endl;
    assert( v7.id == 7 );
    assert( v7.label == "basic_vertex" );

    auto bleh_value_7 = std::any_cast<int>(g->V(7).values("bleh").next());
    std::cout << "property value of bleh on vertex 7 is " << bleh_value_7 << std::endl;
    assert( bleh_value_7 == 66 );

    assert( std::any_cast<size_t>(g->V().values("bleh").count().next()) == 2 );

    std::cout << "testing basic adjacency querying" << std::endl;
    std::vector<uint32_t> ids;
    g->V(0).out().id().forEachRemaining([&ids](std::any& id){
        std::cout << std::any_cast<uint32_t>(id) << " ";
        ids.push_back(std::any_cast<uint32_t>(id));
    });
    std::cout << std::endl;

    assert( ids.size() == 3 && ids[0] == 1 && ids[1] == 2 && ids[2] == 3 );

    assert(
        std::any_cast<size_t>(g->V(7).out().count().next()) == 0
    );

    std::cout << "testing reductions" << std::endl;
    assert(
        std::any_cast<uint32_t>(g->V(0).out().id().min().next()) == 1
    );

    std::cout << "DONE!" << std::endl;
}
