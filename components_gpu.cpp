#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <unordered_set>

#include "traversal/GraphTraversal.h"
#include "structure/Graph.h"
#include "structure/CPUGraph.h"
#include "structure/GPUGraph.h"
#include "util/C.h"
#include "util/gremlin_utils.hpp"

#define LABEL_V "basic_vertex"
#define LABEL_E "basic_edge"
#define NAME "name"

int main(int argc, char* argv[]) {
    CPUGraph graph;
    graph.create_index(VERTEX_INDEX, NAME, [](boost::any& a) { 
            std::hash<std::string> hf;
            return hf(boost::any_cast<std::string>(a));
        }, [](boost::any& a, boost::any& b) {
            std::string c = boost::any_cast<std::string>(a);
            std::string d = boost::any_cast<std::string>(b);
            return c == d;
        });
    GraphTraversalSource* g = graph.traversal();

    std::string filename = std::string(argv[1]);
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
        //std::cout << boost::any_cast<uint64_t>(v1->id()) << " " << boost::any_cast<std::string>(v1->property(NAME)->value()) << "\n";
        
        if(0 == names.count(std::string(id2))) v2 = boost::any_cast<Vertex*>(g->addV(LABEL_V)->property(NAME, std::string(id2))->next());
        else v2 = boost::any_cast<Vertex*>(g->V()->has(NAME, std::string(id2))->next());

        names.insert(std::string(id2));
        //std::cout << boost::any_cast<uint64_t>(v2->id()) << " " << boost::any_cast<std::string>(v2->property(NAME)->value()) << "\n";

        //std::cout << boost::any_cast<uint64_t>(v1->id()) << " - - " << boost::any_cast<uint64_t>(v2->id()) << "\n";
        
        try {
            g->V(v1)->addE(LABEL_E)->to(v2)->iterate();
        } catch(const std::exception& err) {
            std::cout << err.what() << "\n";
            return -1;
        }
    }

    std::cout << "creating gpu graph!" << std::endl;
    cudaSetDevice(0);
    GPUGraph gpu_graph(graph);
    std::cout << "gpu graph created!" << std::endl;
    auto h = g;//gpu_graph.traversal();

    /*
    try {
        h->V()->id()->forEachRemaining([](boost::any& cnt){
            std::cout << boost::any_cast<uint64_t>(cnt) << std::endl;
        });
    } catch(const std::exception& err) {
        std::cout << err.what() << std::endl;
        return -1;
    }
    */

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;
    std::cerr << "Ingest time: " << elapsed.count() << " seconds." << std::endl;

    try {
        start = std::chrono::system_clock::now();
        //g->V()->property("d", __->out()->count())->iterate();
        /*
        std::list<Vertex*> vertices = graph.vertices();
        for(auto it = vertices.begin(); it != vertices.end(); ++it) {
            Vertex* v = *it;
            std::cout << boost::any_cast<size_t>(v->property("d")->value()) << std::endl;
        }*/
        h->V()->property("cc", __->id())->iterate();
        h->V()->property("old_cc", __->values("cc"))->iterate();
        /*
        The old traversal
        for(int k = 0; k < 1; ++k) {
            g->V()->property("cc", __->coalesce({__->both(), __->identity()})->values("cc")->min(C<uint64_t>::compare()))->iterate();
        }
        */
        
        h->V()->repeat(
                __->property("old_cc", __->values("cc"))
                ->property("cc", 
                    __->coalesce({__->both()->values("cc"), __->values("cc")})->min(C<uint64_t>::compare())
                )
        )
        ->until(
            __->valueMap({"cc","old_cc"})->by(__->unfold())
            ->where("cc", P::neq("old_cc"))
            ->count()
            ->is(0)
        )
        ->iterate();

        end = std::chrono::system_clock::now();
        elapsed = end-start;
        std::cerr << "CC 1x time: " << elapsed.count() << " seconds." << std::endl;
        std::unordered_set<int> comp_set;
        h->V()->values("cc")->forEachRemaining([g,&comp_set](boost::any& v) {
            int id = boost::any_cast<uint64_t>(v);
            comp_set.insert(id);
            //std::cout << id << std::endl;
        });
        std::cout << comp_set.size() << " components!" << std::endl;
    } catch(const std::exception& err) {
        std::cout << err.what() << std::endl;
        return -1;
    }

    fclose(f);
}