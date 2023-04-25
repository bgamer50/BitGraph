#include "structure/GPUVertex.cuh"
#include "structure/BitVertex.h"
#include "structure/GPUGraph.cuh"

GPUVertex::GPUVertex(GPUGraph* graph, BitVertex* v, size_t gpu_vertex_id) : GPUVertex(graph, gpu_vertex_id, boost::any_cast<uint64_t>(v->id()), v->label()) {}

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