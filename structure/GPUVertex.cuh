#pragma once

#include "gremlinxx/gremlinxx.h"
#include <string>

class GPUGraph;
class BitVertex;

class GPUVertex : public Vertex {
    private:
        size_t cpu_vertex_id;
        std::string vertex_label;
        GPUGraph* graph;

    public:
        size_t gpu_vertex_id;

        GPUVertex(GPUGraph* graph, size_t gpu_vertex_id, size_t cpu_vertex_id, std::string vertex_label)
        : Vertex() {
            this->graph = graph;
            this->gpu_vertex_id = gpu_vertex_id;
            this->cpu_vertex_id = cpu_vertex_id;
            this->vertex_label = vertex_label;
        }

        /**
            Construct a new GPUVertex from a CPU Vertex.
            The new reference Vertex won't be linked to the original CPU Vertex.
        **/
        GPUVertex(GPUGraph* graph, BitVertex* v, size_t gpu_vertex_id);

        virtual Graph* getGraph();
        virtual boost::any id();
        virtual std::string label();
        virtual std::vector<Edge*> edges(Direction dir);

        using Vertex::property;
        virtual Property* property(std::string key);
        virtual Property* property(Cardinality cardinality, std::string key, boost::any& value);

        using Vertex::properties;
        virtual std::vector<Property*> properties(std::vector<std::string> keys);

};


