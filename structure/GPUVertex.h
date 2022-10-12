#ifndef GPU_VERTEX_H
#define GPU_VERTEX_H

#include "structure/Graph.h"
#include "structure/Vertex.h"
#include "structure/Edge.h"
#include "structure/VertexProperty.h"
#include <string>

class GPUGraph;

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
        GPUVertex(GPUGraph* graph, BitVertex* v, size_t gpu_vertex_id) : GPUVertex(graph, gpu_vertex_id, boost::any_cast<uint64_t>(v->id()), v->label()) {}

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

#include "structure/GPUGraph.h"

Graph* GPUVertex::getGraph() { 
    return static_cast<Graph*>(this->graph); 
}

boost::any GPUVertex::id() { 
    return cpu_vertex_id; 
}

std::string GPUVertex::label() { 
    return this->vertex_label; 
}

std::vector<Edge*> GPUVertex::edges(Direction dir) { 
    // TODO this may change in the future
    throw std::runtime_error("Cannot directly access edges from GPU Vertex!"); 
} 

Property* GPUVertex::property(std::string key) { 
    return new VertexProperty(key, this->graph->get_property(key, this->gpu_vertex_id)); 
}

Property* GPUVertex::property(Cardinality cardinality, std::string key, boost::any& value) {
    if(cardinality != SINGLE) throw std::runtime_error("Only SINGLE cardinality is supported by GPU properties!");
    this->graph->set_property(key, this->gpu_vertex_id, value);
    return new VertexProperty(key, value);
}

std::vector<Property*> GPUVertex::properties(std::vector<std::string> keys) {
    std::vector<Property*> property_values;
    for(std::string property_key : keys) property_values.push_back(this->property(property_key));
    return property_values;
}

#endif