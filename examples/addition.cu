#include <vector>
#include <any>
#include <string>
#include <chrono>
#include <ctime>
#include <unordered_set>

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

}