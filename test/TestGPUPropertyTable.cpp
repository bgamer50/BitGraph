#include "traversal/Comparison.h"
#include "structure/memory/GPUPropertyTable.cuh"

int main(int argc, char* argv[]) {
    try {
        auto ix = bitgraph::memory::GPUPropertyIndex(30, gremlinxx::comparison::C::FLOAT64);
        std::cout << "create index" << std::endl;

        ix.set(100, 3.9);
        std::cout << "set index" << std::endl;

        std::vector<size_t> vals = {100};
        std::cout << boost::any_cast<double>(ix.get(vals)[0]) << std::endl;
    } catch(const std::exception& err) {
        std::cout << err.what() << "\n";
        return -1;
    }
}
